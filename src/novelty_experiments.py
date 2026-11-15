"""Proof that the novelty flag means something.

(a) Hide one class: build the novelty reference WITHOUT big-rock patches, then measure how well the
    novelty score separates true big-rock test pixels (test_min3 expert labels) from all other
    labelled pixels (AUROC, pixel-weighted per 32x32 cell). Compared with the normal reference.
(b) New rover, new place: ~30 full-frame Perseverance Navcam images from NASA's raw-image JSON feed
    (Jezero crater) -> % of patches flagged unknown, vs Curiosity test_min3 images (Gale crater).

Writes results/novelty.json, results/novelty_experiments.png, results/perseverance_images.json.
"""
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.data import load_mask  # noqa: E402
from src.novelty import PCTL, Novelty, build, knn_score  # noqa: E402

FEED = ("https://mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020&feedtype=json"
        "&num=100&page={page}&order=sol+desc&search=NAVCAM_LEFT")
ROCK = C.CLASSES.index("bigrock")
NEAR = int(C.GRID * (1 - C.NEAR_FIELD_FRAC))  # first grid row of the lower 60% (near field)


def cell_counts(mask):
    m = mask.reshape(C.GRID, C.CELL, C.GRID, C.CELL).transpose(0, 2, 1, 3).reshape(C.GRID * C.GRID, -1)
    rock = (m == ROCK).sum(1)
    other = ((m != C.IGNORE) & (m != ROCK)).sum(1)
    return rock, other


def weighted_auroc(scores, pos_w, neg_w):
    from sklearn.metrics import roc_auc_score, roc_curve

    s = np.concatenate([scores, scores])
    y = np.concatenate([np.ones_like(scores), np.zeros_like(scores)])
    w = np.concatenate([pos_w, neg_w]).astype(np.float64)
    keep = w > 0
    fpr, tpr, _ = roc_curve(y[keep], s[keep], sample_weight=w[keep])
    return float(roc_auc_score(y[keep], s[keep], sample_weight=w[keep])), fpr, tpr


def experiment_a(test_ids):
    print("(a) reference without big-rock patches")
    ref_nr, thr_nr, _ = build(exclude_class=ROCK)
    full = Novelty()
    s_nr, s_full, pw, nw = [], [], [], []
    for iid in test_ids:
        g = np.load(C.FEAT_DIR / "grid" / f"{iid}.npy").reshape(-1, C.FEAT_DIM)
        s_nr.append(knn_score(g, ref_nr))
        s_full.append(knn_score(g, full.ref))
        r, o = cell_counts(load_mask("test_min3", iid))
        pw.append(r)
        nw.append(o)
    s_nr, s_full, pw, nw = map(np.concatenate, (s_nr, s_full, pw, nw))
    auc_nr, fpr_nr, tpr_nr = weighted_auroc(s_nr, pw, nw)
    auc_full, fpr_f, tpr_f = weighted_auroc(s_full, pw, nw)
    flag_rock = float((pw * (s_nr > thr_nr)).sum() / pw.sum())
    flag_other = float((nw * (s_nr > thr_nr)).sum() / nw.sum())
    res = {"auroc_bigrock_hidden": auc_nr, "auroc_normal_reference": auc_full,
           "threshold_bigrock_hidden": thr_nr,
           "flagged_bigrock_pixels": flag_rock, "flagged_other_pixels": flag_other,
           "bigrock_pixels": int(pw.sum()), "other_pixels": int(nw.sum()), "test_images": len(test_ids)}
    print(f"  AUROC hidden={auc_nr:.3f} normal={auc_full:.3f}; flagged big-rock {flag_rock:.1%} vs other {flag_other:.1%}")
    return res, (fpr_nr, tpr_nr, fpr_f, tpr_f)


def fetch_perseverance(n=30, max_pages=6):
    """Full-frame, downward-looking left Navcam images (ground in view)."""
    picked, seen_sols = [], {}
    for page in range(max_pages):
        with urllib.request.urlopen(urllib.request.Request(FEED.format(page=page),
                                                           headers={"User-Agent": "sandtrap/1.0"}), timeout=60) as r:
            imgs = json.loads(r.read())["images"]
        for im in imgs:
            ext = im.get("extended") or {}
            dim = ext.get("dimension", "")
            try:
                el = float(ext.get("mastEl", "nan"))
            except ValueError:
                continue
            if dim not in ("(1280,960)", "(1296,976)") or not (el < -15):
                continue
            if seen_sols.get(im["sol"], 0) >= 3:  # spread over many sols / places
                continue
            seen_sols[im["sol"]] = seen_sols.get(im["sol"], 0) + 1
            picked.append({"imageid": im["imageid"], "sol": im["sol"], "site": im.get("site"),
                           "url": im["image_files"]["full_res"], "mastEl": el, "credit": im.get("credit")})
            if len(picked) >= n:
                return picked
    return picked


def to_square_grey(data):
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("L")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((C.ORIG, C.ORIG))
    return np.asarray(img, np.uint8)


def experiment_b(test_ids, nov):
    from src.features import DinoFeatures

    print("(b) Perseverance Navcam (Jezero) vs Curiosity test_min3 (Gale)")
    try:
        picks = fetch_perseverance()
    except Exception as ex:  # noqa: BLE001
        return {"skipped": f"feed unavailable: {ex}"}, None
    if len(picks) < 10:
        return {"skipped": f"only {len(picks)} suitable images"}, None
    cache = C.DATA_DIR / "perseverance"
    cache.mkdir(parents=True, exist_ok=True)
    dino = DinoFeatures()
    per_img, per_near = [], []
    for p in picks:
        f = cache / f"{p['imageid']}.png"
        if not f.exists():
            with urllib.request.urlopen(urllib.request.Request(p["url"], headers={"User-Agent": "sandtrap/1.0"}),
                                        timeout=120) as r:
                grey = to_square_grey(r.read())
            from PIL import Image
            Image.fromarray(grey).save(f)
        grey = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(f).convert("L"))
        u = nov.unknown(dino([grey])[0])
        p["unknown_frac"] = float(u.mean())
        p["unknown_frac_near_field"] = float(u[NEAR:].mean())
        per_img.append(p["unknown_frac"])
        per_near.append(p["unknown_frac_near_field"])
    cur, cur_near = [], []
    for iid in test_ids:
        u = nov.unknown(np.load(C.FEAT_DIR / "grid" / f"{iid}.npy"))
        cur.append(float(u.mean()))
        cur_near.append(float(u[NEAR:].mean()))
    (C.RESULTS_DIR / "perseverance_images.json").write_text(json.dumps(picks, indent=1))
    res = {"feed": FEED.format(page=0), "n_perseverance": len(picks),
           "perseverance_unknown_mean": float(np.mean(per_img)),
           "perseverance_unknown_median": float(np.median(per_img)),
           "curiosity_test_unknown_mean": float(np.mean(cur)),
           "curiosity_test_unknown_median": float(np.median(cur)),
           "n_curiosity": len(cur),
           "perseverance_unknown_near_field_mean": float(np.mean(per_near)),
           "curiosity_test_unknown_near_field_mean": float(np.mean(cur_near)),
           "note": "Perseverance frames centre-cropped to square and converted to grey; different camera and site."}
    print(f"  unknown (whole frame): Perseverance {np.mean(per_img):.1%} vs Curiosity {np.mean(cur):.1%}")
    print(f"  unknown (near field):  Perseverance {np.mean(per_near):.1%} vs Curiosity {np.mean(cur_near):.1%}")
    return res, (per_near, cur_near)


def plot(roc, dist, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 1 + (dist is not None)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    fpr_nr, tpr_nr, fpr_f, tpr_f = roc
    ax.plot(fpr_nr, tpr_nr, color="#7b2cbf", lw=2, label="big rock hidden from reference")
    ax.plot(fpr_f, tpr_f, color="#9aa0a6", lw=2, label="normal reference")
    ax.plot([0, 1], [0, 1], ls=":", color="#bbb", lw=1)
    ax.set_xlabel("other terrain flagged (false-positive rate)")
    ax.set_ylabel("big-rock pixels flagged (true-positive rate)")
    ax.set_title("(a) Can it spot a terrain it never saw?")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if dist is not None:
        ax = axes[1]
        per, cur = dist
        bins = np.linspace(0, 1, 21)
        ax.hist(np.array(cur), bins=bins, color="#9aa0a6", alpha=0.9, label=f"Curiosity test images (n={len(cur)})",
                weights=np.ones(len(cur)) / len(cur))
        ax.hist(np.array(per), bins=bins, color="#7b2cbf", alpha=0.75,
                label=f"Perseverance Navcam (n={len(per)})", weights=np.ones(len(per)) / len(per))
        ax.set_xlabel("share of near-field patches flagged unknown")
        ax.set_ylabel("share of images")
        ax.set_title("(b) New rover, new crater")
        ax.legend(frameon=False, fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    man = json.loads((C.DATA_DIR / "manifest.json").read_text())
    test_ids = man["test_min3"]["ids"]
    nov = Novelty()
    res_a, roc = experiment_a(test_ids)
    res_b, dist = experiment_b(test_ids, nov)
    out = {"method": f"kNN (k=10) mean distance on L2-normalised DINOv2-small patch features; threshold = "
                     f"p{PCTL:.0f} of held-out validation patches",
           "a_hidden_class": res_a, "b_perseverance": res_b}
    (C.RESULTS_DIR / "novelty.json").write_text(json.dumps(out, indent=2))
    plot(roc, dist, C.RESULTS_DIR / "novelty_experiments.png")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
