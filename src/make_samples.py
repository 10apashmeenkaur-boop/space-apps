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
        if lab.mean() < 0.5:
            continue
        frac = np.bincount(near[lab], minlength=C.N_CLASSES)[: C.N_CLASSES] / lab.sum()
        stats.append((iid, frac))

    chosen, plan = [], [("sand", 3), ("bigrock", 2), ("bedrock", 2), ("soil", 1)]
    for cls, k in plan:
        ci = C.CLASSES.index(cls)
        ranked = sorted(stats, key=lambda s: -s[1][ci])
        for iid, frac in ranked:
            if k == 0:
                break
            if iid not in [c[1] for c in chosen]:
                chosen.append((cls, iid, frac))
                k -= 1

    C.SAMPLES_DIR.mkdir(exist_ok=True)
    for f in C.SAMPLES_DIR.glob("*.png"):
        f.unlink()
    for i, (cls, iid, frac) in enumerate(chosen, 1):
        name = f"{i:02d}_{cls}_{iid[:8]}.png"
        Image.fromarray(load_image(iid)).save(C.SAMPLES_DIR / name, optimize=True)
        print(name, " ".join(f"{c}={v:.0%}" for c, v in zip(C.CLASSES, frac)))


if __name__ == "__main__":
    main()
