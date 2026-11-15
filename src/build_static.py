"""Build the free static version of the app into web/ (for a Hugging Face static Space or any web host).

Everything the browser needs is precomputed here from committed repo files (no data/ needed):
  data/hirise_slope.u8, hirise_elev.i16, mola_slope.f32, mola_elev.i16 + grids.json  (route analysis)
  data/layers.json, ai_images.json, ai_stops.json, metrics.json, ai_samples.json
  thumbs/<id>.jpg (photo | AI | ground truth), samples/*.png (AI tab), assets/confusion_matrix.png

Run:  python src/build_static.py [--github URL]
"""
import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

SRC = C.ROOT / "static_site"
OUT = C.ROOT / "web"
RANK = {"GO": 0, "GO WITH CAUTION": 1, "HOLD": 2}

SPACE_README = """---
title: Marswalk Planner - Gale Crater
emoji: 🛰️
colorFrom: red
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
short_description: Layered Marswalk planner for Gale Crater + AI ground checks
---

Static build of the Marswalk Planner (NASA Space Apps 2026, Challenge #10). The map, layers, route planner and
AI photo markers run in the browser; the AI Ground Check tab shows precomputed results of the same model.
Source code, method and results: {github}
"""


def ai_stops(images):
    groups = {}
    for im in images:
        groups.setdefault((im["site"], im["drive"]), []).append(im)
    out = []
    for (site, drive), ims in groups.items():
        ex = [i["expert"] for i in ims if i.get("expert")]
        out.append({"site": site, "drive": drive, "lat": ims[0]["lat"], "lon": ims[0]["lon"], "loc": ims[0]["loc"],
                    "sol": ims[0]["sol_min"], "sol_max": ims[0]["sol_max"], "n": len(ims), "ids": [i["id"] for i in ims],
                    "sand": float(np.mean([i["ai"]["sand"] for i in ims])),
                    "rock": float(np.mean([i["ai"]["bigrock"] for i in ims])),
                    "expert_sand": float(np.mean([e["sand"] for e in ex])) if ex else None,
                    "expert_rock": float(np.mean([e["bigrock"] for e in ex])) if ex else None,
                    "verdict": max((i["verdict"] for i in ims), key=RANK.get)})
    return out


def write_grids(data_dir, github):
    from src.geo import slope_deg

    h = np.load(C.ROOT / "geo" / "hirise_dem20.npz")
    m = np.load(C.ROOT / "geo" / "mola_dem.npz")
    np.ascontiguousarray(h["slope"], dtype=np.uint8).tofile(data_dir / "hirise_slope.u8")
    np.ascontiguousarray(h["elev"], dtype="<i2").tofile(data_dir / "hirise_elev.i16")
    dem = m["dem"].astype(np.float32)
    s, n, w, e = m["bounds"].tolist()
    ppd = dem.shape[0] / (n - s)
    np.ascontiguousarray(slope_deg(dem, s, n, ppd, ppd), dtype="<f4").tofile(data_dir / "mola_slope.f32")
    np.ascontiguousarray(m["dem"], dtype="<i2").tofile(data_dir / "mola_elev.i16")
    meta = {"hirise": {"h": int(h["slope"].shape[0]), "w": int(h["slope"].shape[1]), "bounds": h["bounds"].tolist()},
            "mola": {"h": int(dem.shape[0]), "w": int(dem.shape[1]), "bounds": m["bounds"].tolist()},
            "github": github}
    (data_dir / "grids.json").write_text(json.dumps(meta))


def ai_samples(out_dir):
    """Run the real model on the 8 demo samples and save overlay + briefing (same code as the Gradio app)."""
    from PIL import Image

    from src.briefing import make_briefing
    from src.infer import proba_grid
    from src.planner import plan

    import app  # reuses render_overlay (importing does not launch the server)

    dino, clf = app.models()  # same loaded models as the app (avoids a double import race)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    cards = []
    for p in sorted(C.SAMPLES_DIR.glob("*.png")):
        grey = np.asarray(Image.open(p).convert("L"), np.uint8)
        prob = proba_grid(clf, dino([grey])[0])
        res = plan(prob)
        brief = make_briefing(res)
        over = app.render_overlay(grey, prob, res, view=512)
        g = np.repeat(np.asarray(Image.fromarray(grey).resize((512, 512)))[..., None], 3, -1)
        Image.fromarray(np.concatenate([g, np.full((512, 6, 3), 255, np.uint8), over], 1)).save(
            out_dir / "samples" / f"{p.stem}.jpg", quality=85)
        risk = "—" if brief["risk"] is None else f"{brief['risk']:.2f}"
        body = "".join(f"<li>{html.escape(line)}</li>" for line in brief["lines"])
        cards.append({"name": p.stem, "overlay": f"samples/{p.stem}.jpg", "verdict": brief["verdict"],
                      "briefing": f"<p class='note'>Left: Navcam photo · right: AI terrain map + safe route. "
                                  f"Risk score <b>{risk}</b>.</p><ul>{body}</ul>"})
        print("  sample", p.stem, brief["verdict"])
    return cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", default="https://github.com/10apashmeenkaur-boop/space-apps")
    args = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    data = OUT / "data"
    data.mkdir(parents=True)
    for f in SRC.iterdir():
        shutil.copy(f, OUT / f.name)
    (OUT / "README.md").write_text(SPACE_README.format(github=args.github), encoding="utf-8")

    layers = json.loads((C.ROOT / "geo" / "layers.json").read_text())
    wps = layers["waypoints"]
    layers["waypoints"] = wps[:: max(1, len(wps) // 120)] + [wps[-1]]
    (data / "layers.json").write_text(json.dumps(layers))
    images = json.loads((C.ROOT / "geo" / "ai_images.json").read_text())
    (data / "ai_images.json").write_text(json.dumps(images))
    (data / "ai_stops.json").write_text(json.dumps(ai_stops(images)))
    shutil.copy(C.RESULTS_DIR / "metrics.json", data / "metrics.json")
    write_grids(data, args.github)
    shutil.copytree(C.ROOT / "geo" / "ai_thumbs", OUT / "thumbs")
    (OUT / "assets").mkdir()
    shutil.copy(C.RESULTS_DIR / "confusion_matrix.png", OUT / "assets" / "confusion_matrix.png")
    (data / "ai_samples.json").write_text(json.dumps(ai_samples(OUT)))
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"site/: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
