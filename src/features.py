"""Phase 2: DINOv2-small patch features on CPU.

Own preprocessing (no AutoImageProcessor, which would centre-crop to 224):
grey 1024x1024 -> resize 448x448 -> 3 channels -> ImageNet normalise.
Output grid: 32x32 cells x 384 features, one cell = 32x32 px of the original.

Cache (under DATA_DIR/features):
  grid/<id>.npy    full (32,32,384) float16 grid  - test images
  cells/<id>.npz   X (k,384) float16, y (k,), idx (k,) - sampled train cells
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

warnings.filterwarnings("ignore", message=".*symlinks.*")


class DinoFeatures:
    """Frozen DINOv2-small that returns a (B, 32, 32, 384) patch grid."""

    def __init__(self, threads=None):
        import torch
        from transformers import AutoModel

        self.torch = torch
        torch.set_num_threads(threads or os.cpu_count())
        self.model = AutoModel.from_pretrained(C.DINO_MODEL).eval()
        self.mean = torch.tensor(C.MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(C.STD).view(1, 3, 1, 1)

    def preprocess(self, greys):
        from PIL import Image

        arr = np.stack([
            np.asarray(Image.fromarray(g).resize((C.IMG, C.IMG), Image.BILINEAR), dtype=np.float32)
            for g in greys
        ]) / 255.0
        x = self.torch.from_numpy(arr)[:, None].repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def __call__(self, greys):
        """greys: list of uint8 HxW arrays -> float32 (B, GRID, GRID, FEAT_DIM)."""
        x = self.preprocess(greys)
        with self.torch.inference_mode():
            h = self.model(pixel_values=x).last_hidden_state
        B = len(greys)
        assert tuple(h.shape) == (B, C.GRID * C.GRID + 1, C.FEAT_DIM), h.shape
        return h[:, 1:].reshape(B, C.GRID, C.GRID, C.FEAT_DIM).numpy()


def cell_labels(mask):
    """mask (1024,1024) -> (labels (1024,), valid (1024,)) on the 32x32 grid.

    A cell is valid when >= CELL_MIN_LABELLED of its pixels are labelled; its
    label is the majority class among labelled pixels.
    """
    m = mask.reshape(C.GRID, C.CELL, C.GRID, C.CELL).transpose(0, 2, 1, 3).reshape(C.GRID * C.GRID, -1)
    counts = np.stack([(m == k).sum(1) for k in range(C.N_CLASSES)], 1)
    labelled = counts.sum(1)
    valid = labelled >= C.CELL_MIN_LABELLED * C.CELL * C.CELL
    return counts.argmax(1), valid


def balanced_pick(labels, valid, k, rng):
    """Pick up to k valid cell indices, spreading the budget evenly across classes."""
    idx = np.flatnonzero(valid)
    if len(idx) <= k:
        return idx
    groups = [rng.permutation(idx[labels[idx] == c]) for c in range(C.N_CLASSES)]
    groups = [g for g in groups if len(g)]
    picked, budget = [], k
    while budget > 0 and groups:
        share = max(1, budget // len(groups))
        nxt = []
        for g in groups:
            take = g[:share]
            picked.extend(take.tolist())
            budget -= len(take)
            if len(g) > share:
                nxt.append(g[share:])
            if budget <= 0:
                break
        groups = nxt
    return np.array(sorted(picked[:k]))


def _todo(ids, out_dir, ext):
    return [i for i in ids if not (out_dir / f"{i}{ext}").exists()]


def run(split_ids, mode, extractor, batch=8):
    from tqdm import tqdm

    from src.data import load_image, load_mask

    out_dir = C.FEAT_DIR / ("grid" if mode == "grid" else "cells")
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".npy" if mode == "grid" else ".npz"
    todo = _todo(split_ids, out_dir, ext)
    print(f"  {mode}: {len(split_ids) - len(todo)} cached, {len(todo)} to do")
    for b in tqdm(range(0, len(todo), batch), desc=mode, mininterval=10):
        ids = todo[b: b + batch]
        feats = extractor([load_image(i) for i in ids]).astype(np.float16)
        for iid, f in zip(ids, feats):
            if mode == "grid":
                np.save(out_dir / f"{iid}.npy", f)
            else:
                lab, val = cell_labels(load_mask("train", iid))
                rng = np.random.default_rng([C.SEED, int(iid[:8], 16)])
                sel = balanced_pick(lab, val, C.MAX_CELLS_PER_IMAGE, rng)
                X = f.reshape(-1, C.FEAT_DIM)[sel]
                np.savez(out_dir / f"{iid}.npz", X=X, y=lab[sel].astype(np.uint8), idx=sel.astype(np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    man = json.loads((C.DATA_DIR / "manifest.json").read_text())
    train_ids = man["train"]["ids"]
    test_ids = sorted(set().union(*(man[s]["ids"] for s in C.TEST_SPLITS)))

    ext = DinoFeatures()

    # timing probe on 10 images
    from src.data import load_image

    greys = [load_image(i) for i in train_ids[:10]]
    t = time.time()
    for b in range(0, 10, args.batch):
        ext(greys[b: b + args.batch])
    per = (time.time() - t) / 10
    total = len(_todo(train_ids, C.FEAT_DIR / "cells", ".npz")) + len(_todo(test_ids, C.FEAT_DIR / "grid", ".npy"))
    print(f"Timing: {per:.2f} s/image -> estimated {per * total / 60:.1f} min for {total} images")

    run(test_ids, "grid", ext, args.batch)
    run(train_ids, "cells", ext, args.batch)

    # ---- checks ----
    g = np.load(C.FEAT_DIR / "grid" / f"{test_ids[0]}.npy")
    assert g.shape == (C.GRID, C.GRID, C.FEAT_DIM) and g.dtype == np.float16
    n_cells, cls = 0, np.zeros(C.N_CLASSES, int)
    for iid in train_ids:
        z = np.load(C.FEAT_DIR / "cells" / f"{iid}.npz")
        assert z["X"].shape[1] == C.FEAT_DIM and len(z["X"]) == len(z["y"]) <= C.MAX_CELLS_PER_IMAGE
        n_cells += len(z["y"])
        cls += np.bincount(z["y"], minlength=C.N_CLASSES)
    print(f"Check: grid shape {g.shape} {g.dtype}; {len(test_ids)} test grids")
    print(f"Check: {n_cells} train cells from {len(train_ids)} images; per class " +
          ", ".join(f"{c}={n}" for c, n in zip(C.CLASSES, cls)))
    print(f"Check: re-run todo = {len(_todo(train_ids, C.FEAT_DIR / 'cells', '.npz')) + len(_todo(test_ids, C.FEAT_DIR / 'grid', '.npy'))}")


if __name__ == "__main__":
    main()
