"""Unknown-terrain (novelty) detector on DINOv2 patch features.

Method: k-nearest-neighbour distance (k=10) from each 32x32-pixel patch feature to a reference set of
patches sampled from the TRAIN split (cached features). Features are L2-normalised first.
score = mean distance to the k nearest reference patches. A patch is "unknown" when its score is above
the 99th percentile of scores on held-out patches from the validation split (images never in the
reference set), so ~1% of familiar terrain is flagged by construction.

Build:  python src/novelty.py            -> models/novelty_ref.npz, models/novelty.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

K = 10
N_REF = 10000
PCTL = 99.0
REF_P = C.MODELS_DIR / "novelty_ref.npz"
META_P = C.MODELS_DIR / "novelty.json"


def l2n(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def knn_score(q, ref, k=K, chunk=4096):
    """Mean Euclidean distance from each (normalised) query row to its k nearest reference rows."""
    q = l2n(q)
    rn = (ref * ref).sum(1)
    out = np.empty(len(q), np.float32)
    for i in range(0, len(q), chunk):
        qq = q[i:i + chunk]
        d2 = (qq * qq).sum(1)[:, None] + rn[None, :] - 2 * qq @ ref.T
        part = np.partition(np.maximum(d2, 0), k - 1, axis=1)[:, :k]
        out[i:i + chunk] = np.sqrt(part).mean(1)
    return out


def load_cells(ids, exclude_class=None):
    X, y = [], []
    for iid in ids:
        z = np.load(C.FEAT_DIR / "cells" / f"{iid}.npz")
        X.append(z["X"])
        y.append(z["y"])
    X, y = np.concatenate(X).astype(np.float32), np.concatenate(y)
    if exclude_class is not None:
        keep = y != exclude_class
        X, y = X[keep], y[keep]
    return X, y


def build(exclude_class=None, n_ref=N_REF, seed=C.SEED):
    """Reference set from train-split cells + threshold from validation-split cells."""
    ids = json.loads((C.RESULTS_DIR / "image_ids.json").read_text())
    rng = np.random.default_rng(seed)
    Xtr, _ = load_cells(ids["train"], exclude_class)
    ref = l2n(Xtr[rng.choice(len(Xtr), min(n_ref, len(Xtr)), replace=False)])
    Xva, _ = load_cells(ids["val"], exclude_class)
    Xva = Xva[rng.choice(len(Xva), min(20000, len(Xva)), replace=False)]
    s = knn_score(Xva, ref)
    return ref, float(np.percentile(s, PCTL)), s


class Novelty:
    """Runtime scorer used by the app."""

    def __init__(self):
        z = np.load(REF_P)
        self.ref = z["ref"].astype(np.float32)
        self.threshold = float(json.loads(META_P.read_text())["threshold"])

    def grid(self, feats):
        """(32,32,384) features -> (32,32) novelty scores."""
        return knn_score(feats.reshape(-1, C.FEAT_DIM), self.ref).reshape(C.GRID, C.GRID)

    def unknown(self, feats):
        return self.grid(feats) > self.threshold


def main():
    ref, thr, s = build()
    C.MODELS_DIR.mkdir(exist_ok=True)
    np.savez_compressed(REF_P, ref=ref.astype(np.float16))
    META_P.write_text(json.dumps({"k": K, "n_ref": len(ref), "percentile": PCTL, "threshold": thr,
                                  "features": "DINOv2-small patch, L2-normalised",
                                  "reference": "train split cells", "calibration": "validation split cells"},
                                 indent=2))
    print(f"reference {ref.shape}, threshold (p{PCTL:.0f} on {len(s)} val patches) = {thr:.4f}")


if __name__ == "__main__":
    main()
