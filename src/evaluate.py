"""Phase 3b: pixel-level evaluation on the official AI4Mars test splits.

Probabilities on the 32x32 grid are bilinearly upsampled to 1024x1024; the top
class per pixel is compared with the expert mask, ignoring 255 pixels.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.data import load_image, load_mask  # noqa: E402
from src.infer import load_classifier, pixel_pred, proba_grid  # noqa: E402
from src.train import confusion, iou_from_cm  # noqa: E402


def metrics_from_cm(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    iou = iou_from_cm(cm)
    return {
        "pixel_accuracy": float(tp.sum() / cm.sum()),
        "mIoU": float(np.nanmean(iou)),
        "per_class": {c: {"precision": float(p), "recall": float(r), "IoU": float(i),
                          "support_pixels": int(s)}
                      for c, p, r, i, s in zip(C.CLASSES, prec, rec, iou, cm.sum(1))},
        "confusion": cm.tolist(),
    }


def colorize(lab, grey=None, alpha=0.5):
    pal = np.array([C.COLORS[c] for c in C.CLASSES] + [(0, 0, 0)], dtype=np.float32)
    idx = np.where(lab == C.IGNORE, C.N_CLASSES, lab)
    col = pal[idx]
    if grey is None:
        return col.astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, -1).astype(np.float32)
    out = np.where((lab == C.IGNORE)[..., None], rgb, (1 - alpha) * rgb + alpha * col)
    return out.astype(np.uint8)


def save_example(path, grey, mask, pred, title):
    import matplotlib.pyplot as plt

    shown = pred.copy()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
    ax[0].imshow(grey, cmap="gray")
    ax[0].set_title("Navcam image")
    ax[1].imshow(colorize(mask, grey))
    ax[1].set_title("Expert label (test_min3)")
    ax[2].imshow(colorize(shown, grey))
    ax[2].set_title("Our prediction")
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    man = json.loads((C.DATA_DIR / "manifest.json").read_text())
    ids_used = json.loads((C.RESULTS_DIR / "image_ids.json").read_text())
    train_like = set(ids_used["train"]) | set(ids_used["val"])
    for s in C.TEST_SPLITS:
        overlap = train_like & set(man[s]["ids"])
        assert not overlap, f"{s} overlaps train/val: {len(overlap)}"
    print("Check: no test image id appears in train/val")

    clf = load_classifier()
    info = json.loads((C.MODELS_DIR / "label_info.json").read_text())

    # Majority-class baseline from TRAIN pixel counts (no test information used)
    majority = int(np.argmax(man["train"]["class_pixels"]))

    test_ids = sorted(set().union(*(man[s]["ids"] for s in C.TEST_SPLITS)))
    preds = {}
    for i, iid in enumerate(test_ids):
        g = np.load(C.FEAT_DIR / "grid" / f"{iid}.npy")
        preds[iid] = pixel_pred(proba_grid(clf, g))
        if (i + 1) % 100 == 0:
            print(f"  predicted {i + 1}/{len(test_ids)}")

    results = {"model": info["chosen"], "baseline_class": C.CLASSES[majority], "splits": {}}
    per_image = {}
    for s in C.TEST_SPLITS:
        cm = np.zeros((C.N_CLASSES, C.N_CLASSES), np.int64)
        cm_base = np.zeros_like(cm)
        for iid in man[s]["ids"]:
            m = load_mask(s, iid)
            lab = m != C.IGNORE
            if not lab.any():
                continue
            y, p = m[lab].astype(np.int64), preds[iid][lab].astype(np.int64)
            c = confusion(y, p)
            cm += c
            cm_base += confusion(y, np.full_like(y, majority))
            if s == "test_min3":
                per_image[iid] = (float(np.trace(c) / c.sum()), float(lab.mean()))
        results["splits"][s] = {"n_images": len(man[s]["ids"]), "model": metrics_from_cm(cm),
                                "baseline": metrics_from_cm(cm_base)}

    # ---- table ----
    print(f"\nModel: {info['chosen']}   baseline = always '{C.CLASSES[majority]}'")
    hdr = f"{'split':10s} {'acc':>6s} {'mIoU':>6s} | " + " ".join(f"{c[:7]+' R':>9s} {c[:7]+' IoU':>11s}" for c in C.CLASSES) + " | base acc  base mIoU"
    print(hdr)
    for s, r in results["splits"].items():
        m, b = r["model"], r["baseline"]
        print(f"{s:10s} {m['pixel_accuracy']:6.3f} {m['mIoU']:6.3f} | " +
              " ".join(f"{m['per_class'][c]['recall']:9.3f} {m['per_class'][c]['IoU']:11.3f}" for c in C.CLASSES) +
              f" | {b['pixel_accuracy']:8.3f}  {b['mIoU']:9.3f}")

    out = C.RESULTS_DIR
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    # ---- confusion matrix (test_min3, row-normalised) ----
    cm = np.array(results["splits"]["test_min3"]["model"]["confusion"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(C.N_CLASSES):
        for j in range(C.N_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=11)
    ax.set_xticks(range(C.N_CLASSES), C.CLASSES)
    ax.set_yticks(range(C.N_CLASSES), C.CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expert label")
    ax.set_title("test_min3 confusion (row = recall)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    # ---- 3 good + 3 failure examples (images with >=20% labelled pixels) ----
    ex = out / "examples"
    ex.mkdir(parents=True, exist_ok=True)
    for f in ex.glob("*.png"):
        f.unlink()
    ranked = sorted([(acc, iid) for iid, (acc, frac) in per_image.items() if frac >= 0.20])
    picks = [("good", r) for r in ranked[-3:][::-1]] + [("fail", r) for r in ranked[:3]]
    for k, (kind, (acc, iid)) in enumerate(picks):
        save_example(ex / f"{kind}_{k % 3 + 1}_{iid}.png", load_image(iid), load_mask("test_min3", iid),
                     preds[iid], f"{kind.upper()}: pixel accuracy {acc:.1%}")
    print(f"\nSaved metrics.json, confusion_matrix.png, {len(picks)} examples -> {out}")


if __name__ == "__main__":
    main()
