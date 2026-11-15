"""Geolocate AI4Mars test images on Curiosity's traverse and attach AI ground-check results.

Navcam product names encode the rover's location counter (RMC):
    NLB_537932733EDR_F0600888NCAM00320M1.JPG
         SCLK      ^ F + site (3 digits) + drive (4 digits)
The rover position for each site/drive comes from NASA MMGIS waypoints (MSL_waypoints.json):
an exact site/drive match is used directly; otherwise the position is interpolated between the
nearest waypoints of the same site by drive number (approximate).

Filenames are not a dataset column, but the parquet image structs keep them; images are matched to
our ids through the same pixel hash used in src/data.py.

Writes geo/ai_images.json and geo/ai_thumbs/<id>.jpg.   Run: python src/geolocate.py
"""
import io
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.briefing import make_briefing  # noqa: E402
from src.data import _stream, image_id, load_image, load_mask, to_grey  # noqa: E402
from src.infer import load_classifier, proba_grid, upsample  # noqa: E402
from src.planner import plan  # noqa: E402

GEO = C.ROOT / "geo"
NAME_RE = re.compile(r"^(N[LR][AB])_(\d{9})EDR_F(\d{3})(\d{4})NCAM", re.I)


def parse_name(name):
    m = NAME_RE.match(Path(name).name)
    if not m:
        return None
    return {"camera": m[1].upper(), "sclk": int(m[2]), "site": int(m[3]), "drive": int(m[4])}


def locate(site, drive, waypoints):
    """(lat, lon, quality, sol_range) for a site/drive from MMGIS waypoints."""
    same = sorted((w for w in waypoints if w["site"] == site), key=lambda w: w["drive"])
    if not same:
        return None
    for w in same:
        if w["drive"] == drive:
            return w["lat"], w["lon"], "exact", (w["sol"], w["sol"])
    lo = [w for w in same if w["drive"] < drive]
    hi = [w for w in same if w["drive"] > drive]
    if lo and hi:
        a, b = lo[-1], hi[0]
        t = (drive - a["drive"]) / (b["drive"] - a["drive"])
        return (a["lat"] + t * (b["lat"] - a["lat"]), a["lon"] + t * (b["lon"] - a["lon"]),
                "interpolated", (a["sol"], b["sol"]))
    w = (lo or hi)[-1 if lo else 0]
    return w["lat"], w["lon"], "nearest waypoint in site", (w["sol"], w["sol"])


def filenames_by_id(split="test_min3"):
    out = {}
    for row in _stream(split=split):
        v = row["image"]
        img = Image.open(io.BytesIO(v["bytes"]))
        out[image_id(to_grey(img))] = Path(v["path"]).name
    return out


def near_fracs(lab, top_frac=C.NEAR_FIELD_FRAC):
    near = lab[int(lab.shape[0] * (1 - top_frac)):]
    ok = near != C.IGNORE
    if not ok.any():
        return None
    f = np.bincount(near[ok], minlength=C.N_CLASSES)[: C.N_CLASSES] / ok.sum()
    return {c: round(float(v), 3) for c, v in zip(C.CLASSES, f)}


def thumb(grey, prob, path, size=168):
    pal = np.array([C.COLORS[c] for c in C.CLASSES], np.float32)
    g = cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)
    lab = upsample(prob, size).argmax(-1)
    over = (0.5 * np.repeat(g[..., None].astype(np.float32), 3, -1) + 0.5 * pal[lab]).astype(np.uint8)
    side = np.concatenate([np.repeat(g[..., None], 3, -1), np.full((size, 4, 3), 255, np.uint8), over], 1)
    Image.fromarray(side).save(path, quality=80)


def main():
    man = json.loads((C.DATA_DIR / "manifest.json").read_text())
    layers = json.loads((GEO / "layers.json").read_text())
    wps = layers["waypoints"]

    names_p = C.DATA_DIR / "test_filenames.json"
    if names_p.exists():
        names = json.loads(names_p.read_text())
    else:
        names = filenames_by_id()
        names_p.write_text(json.dumps(names, indent=1))
    ids = man["test_min3"]["ids"]
    print(f"filenames recovered: {sum(i in names for i in ids)}/{len(ids)}")

    clf = load_classifier()
    tdir = GEO / "ai_thumbs"
    tdir.mkdir(exist_ok=True)
    for f in tdir.glob("*.jpg"):
        f.unlink()
    out, stats = [], {"exact": 0, "interpolated": 0, "nearest waypoint in site": 0, "unlocated": 0}
    for iid in ids:
        name = names.get(iid)
        meta = parse_name(name) if name else None
        loc = locate(meta["site"], meta["drive"], wps) if meta else None
        if not loc:
            stats["unlocated"] += 1
            continue
        lat, lon, quality, sols = loc
        stats[quality] += 1
        grey = load_image(iid)
        prob = proba_grid(clf, np.load(C.FEAT_DIR / "grid" / f"{iid}.npy"))
        brief = make_briefing(plan(prob))
        thumb(grey, prob, tdir / f"{iid}.jpg")
        out.append({
            "id": iid, "file": name, "camera": meta["camera"], "site": meta["site"], "drive": meta["drive"],
            "sol_min": sols[0], "sol_max": sols[1], "lat": round(lat, 6), "lon": round(lon, 6),
            "loc": quality, "verdict": brief["verdict"],
            "ai": {k: round(v, 3) for k, v in brief["near_field_fractions"].items()},
            "expert": near_fracs(load_mask("test_min3", iid)),
        })
    (GEO / "ai_images.json").write_text(json.dumps(out))
    stops = {(o["site"], o["drive"]) for o in out}
    print(f"located {len(out)} images at {len(stops)} rover stops: {stats}")
    print(f"sols covered: {min(o['sol_min'] for o in out)}..{max(o['sol_max'] for o in out)}")
    agree = [abs(o["ai"]["sand"] - o["expert"]["sand"]) for o in out if o["expert"]]
    print(f"near-field sand %, AI vs experts: mean abs diff {100 * np.mean(agree):.1f} points")


if __name__ == "__main__":
    main()
