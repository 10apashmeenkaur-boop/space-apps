"""Copy 8 varied images from the official test set (test_min3) into samples/ for the demo."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.data import load_image, load_mask  # noqa: E402


def main():
    man = json.loads((C.DATA_DIR / "manifest.json").read_text())
    stats = []
    for iid in man["test_min3"]["ids"]:
        m = load_mask("test_min3", iid)
        near = m[int(C.ORIG * (1 - C.NEAR_FIELD_FRAC)):]
        lab = near != C.IGNORE
        if lab.mean() < 0.3:
            continue
        frac = np.bincount(near[lab], minlength=C.N_CLASSES)[: C.N_CLASSES] / lab.sum()
        stats.append((iid, frac))

    # Prefer mixed scenes (so the route has something to avoid): score = class entropy,
    # with quotas for sand-mixed, big-rock, bedrock-mixed and a clean soil scene.
    def entropy(f):
        f = f[f > 0]
        return float(-(f * np.log(f)).sum())

    S, R, B, O = (C.CLASSES.index(c) for c in ("sand", "bigrock", "bedrock", "soil"))
    rules = [
        ("sand", 4, lambda f: 0.15 <= f[S] <= 0.75),
        ("bigrock", 1, lambda f: f[R] > 0.005),
        ("bedrock", 2, lambda f: 0.3 <= f[B] <= 0.9),
        ("soil", 1, lambda f: f[O] > 0.8),
    ]
    chosen = []
    for cls, k, ok in rules:
        pool = sorted([s for s in stats if ok(s[1]) and s[0] not in [c[1] for c in chosen]],
                      key=lambda s: -(entropy(s[1]) if cls != "soil" else s[1][O]))
        chosen += [(cls, iid, frac) for iid, frac in pool[:k]]
    rest = sorted([s for s in stats if s[0] not in [c[1] for c in chosen]], key=lambda s: -entropy(s[1]))
    chosen += [("mixed", iid, frac) for iid, frac in rest[: max(0, 8 - len(chosen))]]

    C.SAMPLES_DIR.mkdir(exist_ok=True)
    for f in C.SAMPLES_DIR.glob("*.png"):
        f.unlink()
    for i, (cls, iid, frac) in enumerate(chosen, 1):
        name = f"{i:02d}_{cls}_{iid[:8]}.png"
        Image.fromarray(load_image(iid)).save(C.SAMPLES_DIR / name, optimize=True)
        print(name, " ".join(f"{c}={v:.0%}" for c, v in zip(C.CLASSES, frac)))
    build_gallery(chosen)


def build_gallery(chosen):
    """Analysed overlays (AI terrain map + safe route) for the map tab's ground-truth gallery."""
    import cv2

    from src.briefing import make_briefing
    from src.infer import load_classifier, proba_grid, upsample
    from src.planner import plan

    clf = load_classifier()
    pal = np.array([C.COLORS[c] for c in C.CLASSES], np.float32)
    out = C.SAMPLES_DIR / "analysed"
    out.mkdir(exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    index = []
    for i, (_, iid, _) in enumerate(chosen, 1):
        grey = load_image(iid)
        prob = proba_grid(clf, np.load(C.FEAT_DIR / "grid" / f"{iid}.npy"))
        res = plan(prob)
        brief = make_briefing(res)
        v = 512
        g = cv2.resize(grey, (v, v), interpolation=cv2.INTER_AREA).astype(np.float32)
        lab = upsample(prob, v).argmax(-1)
        img = (0.5 * np.repeat(g[..., None], 3, -1) + 0.5 * pal[lab]).astype(np.uint8)
        if res["path"]:
            s = v / res["prob"].shape[0]
            pts = np.array([(int((c + 0.5) * s), int((r + 0.5) * s)) for r, c in res["path"]], np.int32)
            cv2.polylines(img, [pts], False, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, (40, 230, 90), 4, cv2.LINE_AA)
        side = np.concatenate([np.repeat(cv2.resize(grey, (v, v))[..., None], 3, -1), img], 1)
        name = f"{i:02d}_{iid[:8]}.jpg"
        Image.fromarray(side).save(out / name, quality=88)
        f = brief["near_field_fractions"]
        index.append({"file": name, "id": iid,
                      "caption": f"{brief['verdict']} - near field: sand {f['sand']:.0%}, "
                                 f"big rock {f['bigrock']:.0%}, bedrock {f['bedrock']:.0%}, soil {f['soil']:.0%}"})
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"gallery: {len(index)} analysed images -> {out}")


if __name__ == "__main__":
    main()
