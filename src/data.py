"""Phase 1: stream AI4Mars from Hugging Face, clean masks, save per-image npz files.

Layout under DATA_DIR:
  images/<id>.npz           grey image (uint8, 1024x1024), stored once per unique image
  masks/<split>/<id>.npz    cleaned label mask (uint8, values 0-3 or 255)
  manifest.json             image ids per split + class pixel counts

Image id = first 16 hex chars of the MD5 of the raw pixels. The dataset has no
filename column, so a content hash is the only stable id - and it also catches
test images that appear again inside the train split.
"""
import argparse
import hashlib
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

warnings.filterwarnings("ignore", message=".*symlinks.*")

VALID = {0, 1, 2, 3, C.IGNORE}


def to_grey(img) -> np.ndarray:
    """PIL image (any mode) -> uint8 HxW."""
    return np.array(img.convert("L"), dtype=np.uint8)


def clean_mask(label_mask, rover_mask=None, range_mask=None) -> np.ndarray:
    """Label mask -> uint8 HxW with values in {0,1,2,3,255}.

    Rover body and ground beyond 30 m are forced to IGNORE.
    """
    m = np.array(label_mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.uint8)
    for extra in (rover_mask, range_mask):
        if extra is not None:
            e = np.array(extra)
            if e.ndim == 3:
                e = e[..., 0]
            if e.shape == m.shape:
                m[e > 0] = C.IGNORE
    bad = set(np.unique(m).tolist()) - VALID
    assert not bad, f"unexpected label values {bad}"
    return m


def image_id(grey: np.ndarray) -> str:
    return hashlib.md5(grey.tobytes()).hexdigest()[:16]


def labelled_frac(mask: np.ndarray) -> float:
    return float((mask != C.IGNORE).mean())


def class_counts(mask: np.ndarray) -> np.ndarray:
    return np.bincount(mask[mask != C.IGNORE].ravel(), minlength=C.N_CLASSES)[: C.N_CLASSES]


def img_path(iid):
    return C.DATA_DIR / "images" / f"{iid}.npz"


def mask_path(split, iid):
    return C.DATA_DIR / "masks" / split / f"{iid}.npz"


def save_image(iid, grey):
    p = img_path(iid)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, image=grey)


def save_mask(split, iid, mask):
    p = mask_path(split, iid)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, mask=mask)


def load_image(iid) -> np.ndarray:
    return np.load(img_path(iid))["image"]


def load_mask(split, iid) -> np.ndarray:
    return np.load(mask_path(split, iid))["mask"]


def _stream(split=None, data_files=None):
    from datasets import load_dataset

    if data_files:
        return load_dataset(C.HF_DATASET, data_files={"train": data_files}, split="train", streaming=True)
    return load_dataset(C.HF_DATASET, split=split, streaming=True)


def fetch_test_split(split):
    ids, counts = [], np.zeros(C.N_CLASSES, np.int64)
    for row in _stream(split=split):
        grey = to_grey(row["image"])
        mask = clean_mask(row["label_mask"], row.get("rover_mask"), row.get("range_mask"))
        iid = image_id(grey)
        save_image(iid, grey)
        save_mask(split, iid, mask)
        ids.append(iid)
        counts += class_counts(mask)
    print(f"  {split}: {len(ids)} images")
    return ids, counts


def fetch_train_shard(shard, quota, exclude):
    """Take up to `quota` usable labelled images from one train parquet shard."""
    f = f"data/train-{shard:05d}-of-{C.N_TRAIN_SHARDS:05d}.parquet"
    ids, counts, skipped = [], np.zeros(C.N_CLASSES, np.int64), {"few_labels": 0, "in_test": 0}
    for row in _stream(data_files=f):
        if not row["has_labels"] or row["label_mask"] is None:
            continue
        mask = clean_mask(row["label_mask"], row.get("rover_mask"), row.get("range_mask"))
        if labelled_frac(mask) < C.MIN_LABELLED_FRAC:
            skipped["few_labels"] += 1
            continue
        grey = to_grey(row["image"])
        iid = image_id(grey)
        if iid in exclude:
            skipped["in_test"] += 1
            continue
        if iid in ids:
            continue
        save_image(iid, grey)
        save_mask("train", iid, mask)
        ids.append(iid)
        counts += class_counts(mask)
        if len(ids) >= quota:
            break
    print(f"  shard {shard:2d}: kept {len(ids)}  skipped {skipped}", flush=True)
    return ids, counts, skipped


def save_previews(split, ids, n=5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = C.RESULTS_DIR / "previews"
    out.mkdir(parents=True, exist_ok=True)
    pal = np.array([C.COLORS[c] for c in C.CLASSES], dtype=np.uint8)
    for iid in ids[:n]:
        g, m = load_image(iid), load_mask(split, iid)
        rgb = np.stack([g] * 3, -1).astype(np.float32)
        col = rgb.copy()
        lab = m != C.IGNORE
        col[lab] = 0.5 * rgb[lab] + 0.5 * pal[m[lab]]
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(g, cmap="gray")
        ax[0].set_title("image")
        ax[1].imshow(col.astype(np.uint8))
        ax[1].set_title("labels (unlabelled = plain)")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{split}_{iid}.png", dpi=80)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=C.N_TRAIN_IMAGES)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    manifest_p = C.DATA_DIR / "manifest.json"
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}

    print("Test splits")
    for split in C.TEST_SPLITS:
        if split in manifest:
            print(f"  {split}: cached ({len(manifest[split]['ids'])})")
            continue
        ids, counts = fetch_test_split(split)
        manifest[split] = {"ids": ids, "class_pixels": counts.tolist()}
        manifest_p.write_text(json.dumps(manifest))

    test_ids = set().union(*(manifest[s]["ids"] for s in C.TEST_SPLITS))
    print(f"  unique test images: {len(test_ids)}")

    if "train" not in manifest or len(manifest["train"]["ids"]) < args.n_train:
        print("Train (spread over all shards for sol diversity)")
        quota = int(np.ceil(args.n_train / C.N_TRAIN_SHARDS))
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda s: fetch_train_shard(s, quota, test_ids), range(C.N_TRAIN_SHARDS)))
        ids, counts, skipped = [], np.zeros(C.N_CLASSES, np.int64), {"few_labels": 0, "in_test": 0}
        for i, c, s in res:
            for iid in i:
                if iid not in ids:
                    ids.append(iid)
            counts += c
            for k in skipped:
                skipped[k] += s[k]
        ids = ids[: args.n_train]
        manifest["train"] = {"ids": ids, "class_pixels": counts.tolist(), "skipped": skipped}
        manifest_p.write_text(json.dumps(manifest))
    else:
        print(f"Train: cached ({len(manifest['train']['ids'])})")

    # ---- checks ----
    assert not (set(manifest["train"]["ids"]) & test_ids), "train/test overlap!"
    print("\nCheck: train/test id overlap = 0")
    for split in ["train"] + C.TEST_SPLITS:
        cp = np.array(manifest[split]["class_pixels"], dtype=np.float64)
        pct = 100 * cp / cp.sum()
        print(f"  {split:10s} n={len(manifest[split]['ids']):5d}  " +
              "  ".join(f"{c}={p:5.1f}%" for c, p in zip(C.CLASSES, pct)))
    # re-verify mask values on a sample of saved files
    for split in ["train"] + C.TEST_SPLITS:
        for iid in manifest[split]["ids"][:50]:
            assert set(np.unique(load_mask(split, iid)).tolist()) <= VALID
    print("Check: mask values within {0,1,2,3,255}")

    save_previews("train", manifest["train"]["ids"], 3)
    save_previews("test_min3", manifest["test_min3"]["ids"], 2)
    print(f"Previews -> {C.RESULTS_DIR / 'previews'}")


if __name__ == "__main__":
    main()
