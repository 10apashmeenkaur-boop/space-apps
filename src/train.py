"""Phase 3a: train Logistic Regression and LightGBM on DINOv2 cell features.

Validation split is by image (never by cell). The model with the better
validation mIoU is saved to models/classifier.joblib.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def load_cells(ids):
    Xs, ys = [], []
    for iid in ids:
        z = np.load(C.FEAT_DIR / "cells" / f"{iid}.npz")
        Xs.append(z["X"])
        ys.append(z["y"])
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


def confusion(y, p, n=C.N_CLASSES):
    return np.bincount(n * y + p, minlength=n * n).reshape(n, n)


def iou_from_cm(cm):
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1) - tp
    return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)


def main():
    man = json.loads((C.DATA_DIR / "manifest.json").read_text())
    ids = np.array(man["train"]["ids"])
    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(len(ids))
    n_val = int(round(C.VAL_FRAC * len(ids)))
    val_ids, tr_ids = sorted(ids[perm[:n_val]]), sorted(ids[perm[n_val:]])
    assert not set(val_ids) & set(tr_ids)

    Xtr, ytr = load_cells(tr_ids)
    Xva, yva = load_cells(val_ids)
    print(f"train cells {len(ytr)} from {len(tr_ids)} images | val cells {len(yva)} from {len(val_ids)} images")
    print("train class counts", np.bincount(ytr, minlength=C.N_CLASSES).tolist())

    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    candidates = {}

    t = time.time()
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=C.SEED),
    )
    logreg.fit(Xtr, ytr)
    candidates["logreg"] = (logreg, time.time() - t)

    t = time.time()
    lgbm = LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
        class_weight="balanced", random_state=C.SEED, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[early_stopping(40, verbose=False), log_evaluation(0)])
    candidates["lightgbm"] = (lgbm, time.time() - t)

    report = {"n_train_images": len(tr_ids), "n_val_images": len(val_ids),
              "n_train_cells": int(len(ytr)), "n_val_cells": int(len(yva)), "models": {}}
    best, best_miou = None, -1
    for name, (m, secs) in candidates.items():
        p = m.predict(Xva)
        cm = confusion(yva, p)
        iou = iou_from_cm(cm)
        miou = float(np.nanmean(iou))
        acc = float(np.trace(cm) / cm.sum())
        recall = (np.diag(cm) / np.maximum(cm.sum(1), 1)).tolist()
        report["models"][name] = {"val_cell_acc": acc, "val_mIoU": miou,
                                  "val_IoU": dict(zip(C.CLASSES, iou.tolist())),
                                  "val_recall": dict(zip(C.CLASSES, recall)), "fit_seconds": secs}
        print(f"{name:9s} val acc {acc:.3f}  mIoU {miou:.3f}  IoU " +
              " ".join(f"{c}={v:.2f}" for c, v in zip(C.CLASSES, iou)) + f"  ({secs:.0f}s)")
        if miou > best_miou:
            best, best_miou = name, miou

    report["chosen"] = best
    print(f"Chosen: {best}")
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(candidates[best][0], C.MODELS_DIR / "classifier.joblib", compress=3)
    (C.MODELS_DIR / "label_info.json").write_text(json.dumps(
        {"classes": C.CLASSES, "ignore": C.IGNORE, "dino_model": C.DINO_MODEL, "img": C.IMG,
         "grid": C.GRID, "chosen": best, "train": report}, indent=2))
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "image_ids.json").write_text(json.dumps(
        {"train": list(tr_ids), "val": list(val_ids),
         **{s: man[s]["ids"] for s in C.TEST_SPLITS}}, indent=1))


if __name__ == "__main__":
    main()
