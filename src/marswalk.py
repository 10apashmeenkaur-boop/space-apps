"""Marswalk route analysis on the Gale Crater layers built by src/geo.py and src/geolocate.py.

For a list of waypoints (lat, lon) it returns distance, elevation profile, slope risk (HiRISE 1 m DEM
inside its box - slope at a 4 m baseline, max per 20 m cell - and MOLA 463 m outside it, with the source
labelled per leg), AI sand/rock risk from the nearest geolocated AI4Mars Navcam photos, how much of the
route follows ground Curiosity already drove, science stops nearby and an estimated walking time.
"""
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

GEO = C.ROOT / "geo"
MARS_R = 3396190.0
SAMPLE_M = 20.0          # densify the route every 20 m (= DEM cell)
STOP_RADIUS_M = 1500.0   # science stops "near" the route
AI_RADIUS_M = 200.0      # analysed photos considered "near" the route
AI_SHOW = 5              # nearest photos listed in the report
PROVEN_M = 100.0         # within 100 m of Curiosity's traverse = rover-proven ground
FLAT_KMH = 3.0           # assumed suited walking pace on flat ground
STOP_MIN = 15            # minutes per science stop


@lru_cache(maxsize=1)
def layers():
    return json.loads((GEO / "layers.json").read_text())


@lru_cache(maxsize=1)
def ai_images():
    p = GEO / "ai_images.json"
    return json.loads(p.read_text()) if p.exists() else []


@lru_cache(maxsize=1)
def dem():
    """HiRISE-derived 20 m grids: max slope (deg) and mean elevation (m), rows north -> south."""
    z = np.load(GEO / "hirise_dem20.npz")
    slope, elev = z["slope"].astype(np.float32), z["elev"].astype(np.float32)
    s, n, w, e = z["bounds"].tolist()
    return slope, elev, (s, n, w, e)


@lru_cache(maxsize=1)
def mola():
    """MOLA 463 m elevation + slope for the Gale window (fallback outside the HiRISE box)."""
    from src.geo import slope_deg

    z = np.load(GEO / "mola_dem.npz")
    elev = z["dem"].astype(np.float32)
    s, n, w, e = z["bounds"].tolist()
    ppd = elev.shape[0] / (n - s)
    return slope_deg(elev, s, n, ppd, ppd), elev, (s, n, w, e)


@lru_cache(maxsize=1)
def traverse_xy():
    return np.array(layers()["traverse"], np.float64)


def dist_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * MARS_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def densify(points, step=SAMPLE_M):
    out = [points[0]]
    for (a, b), (c, d) in zip(points[:-1], points[1:]):
        n = max(1, int(math.ceil(dist_m(a, b, c, d) / step)))
        for k in range(1, n + 1):
            out.append((a + (c - a) * k / n, b + (d - b) * k / n))
    return np.array(out)


def sample_grid(grid, bounds, lat, lon):
    s, n, w, e = bounds
    h, wd = grid.shape
    r = np.clip(((n - lat) / (n - s) * h).astype(int), 0, h - 1)
    c = np.clip(((lon - w) / (e - w) * wd).astype(int), 0, wd - 1)
    inside = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
    return grid[r, c], inside


def tobler_kmh(grade):
    """Tobler's hiking function, rescaled so flat ground = FLAT_KMH."""
    return FLAT_KMH * np.exp(-3.5 * np.abs(grade + 0.05)) / math.exp(-3.5 * 0.05)


def slope_level(frac15, frac25):
    if frac25 >= 0.02 or frac15 >= 0.20:
        return "HIGH"
    if frac25 > 0 or frac15 >= 0.05:
        return "MEDIUM"
    return "LOW"


def ai_near_route(lat, lon):
    """Analysed AI4Mars photos within AI_RADIUS_M of the route, nearest first."""
    hits = []
    for im in ai_images():
        if abs(im["lat"] - lat).min() > 0.005 or abs(im["lon"] - lon).min() > 0.005:
            continue
        d = float(dist_m(im["lat"], im["lon"], lat, lon).min())
        if d <= AI_RADIUS_M:
            hits.append({**im, "distance_m": d})
    return sorted(hits, key=lambda h: h["distance_m"])


def sample_terrain(lat, lon):
    """Slope, elevation and source ('HiRISE' / 'MOLA' / '') at each point."""
    hs, he, hb = dem()
    ms, me, mb = mola()
    slp_h, in_h = sample_grid(hs, hb, lat, lon)
    elv_h, _ = sample_grid(he, hb, lat, lon)
    slp_m, in_m = sample_grid(ms, mb, lat, lon)
    elv_m, _ = sample_grid(me, mb, lat, lon)
    src = np.where(in_h, "HiRISE", np.where(in_m, "MOLA", ""))
    return np.where(in_h, slp_h, slp_m), np.where(in_h, elv_h, elv_m), src


def analyse(points):
    """points: list of [lat, lon] (>= 2). Returns a dict of route facts."""
    pts = [(float(a), float(b)) for a, b in points]
    if len(pts) < 2:
        return {"error": "Click at least two waypoints on the map."}
    P = densify(pts)
    lat, lon = P[:, 0], P[:, 1]
    slp, elev, src = sample_terrain(lat, lon)
    if (src == "").any():
        return {"error": "Part of the route is outside the Gale Crater data window (135.5–140°E, 3–7.5°S)."}
    seg = dist_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    total = float(seg.sum())
    same = src[:-1] == src[1:]
    dz = np.where(same, np.diff(elev), 0.0)   # no fake climb where the DEM source switches
    grade = dz / np.maximum(seg, 1)
    mid = np.maximum(slp[:-1], slp[1:])
    seg_src = np.where((src[:-1] == "HiRISE") & (src[1:] == "HiRISE"), "HiRISE", "MOLA")
    hours_walk = float(np.sum(seg / 1000 / tobler_kmh(grade)))

    # per leg (waypoint i -> i+1): length, DEM source(s), steepest cell
    legs, k = [], 0
    for i, ((a, b), (c, d)) in enumerate(zip(pts[:-1], pts[1:])):
        n = max(1, int(math.ceil(dist_m(a, b, c, d) / SAMPLE_M)))
        sl = slice(k, k + n)
        length = float(seg[sl].sum())
        hi = float(seg[sl][seg_src[sl] == "HiRISE"].sum())
        frac = hi / length if length else 0.0
        legs.append({"leg": i + 1, "km": length / 1000, "hirise_frac": frac,
                     "source": "HiRISE 1 m" if frac > 0.999 else ("MOLA 463 m" if frac < 0.001 else "mixed"),
                     "max_slope": float(mid[sl].max())})
        k += n

    T = traverse_xy()
    near = []
    for a, b in P[:: max(1, len(P) // 400)]:
        box = (np.abs(T[:, 0] - a) < 0.01) & (np.abs(T[:, 1] - b) < 0.01)
        near.append(dist_m(a, b, T[box, 0], T[box, 1]).min() if box.any() else 1e9)
    proven = float(np.mean(np.array(near) <= PROVEN_M))

    stops = []
    for st in layers()["stops"]:
        d = float(dist_m(st["lat"], st["lon"], lat, lon).min())
        if d <= STOP_RADIUS_M:
            stops.append({**st, "distance_m": d})
    stops.sort(key=lambda s: s["distance_m"])
    science = [s for s in stops if s["kind"] != "Rover position"]

    frac15 = float(np.sum(seg[mid > 15]) / total)
    frac25 = float(np.sum(seg[mid > 25]) / total)
    hi_km = float(seg[seg_src == "HiRISE"].sum()) / 1000
    step = max(1, len(P) // 120)
    return {
        "waypoints": pts,
        "distance_km": total / 1000, "hirise_km": hi_km, "mola_km": total / 1000 - hi_km, "legs": legs,
        "elev_start": float(elev[0]), "elev_end": float(elev[-1]),
        "climb_m": float(np.clip(dz, 0, None).sum()), "descent_m": float(np.clip(-dz, 0, None).sum()),
        "max_slope_deg": float(slp.max()), "median_slope_deg": float(np.median(slp)),
        "frac_gt15": frac15, "frac_gt25": frac25, "slope_risk": slope_level(frac15, frac25),
        "proven_frac": proven,
        "ai": ai_near_route(lat, lon),
        "stops": stops,
        "walk_hours": hours_walk,
        "total_hours": hours_walk + STOP_MIN * len(science) / 60,
        "profile": {"km": np.concatenate([[0], np.cumsum(seg)])[::step].tolist(), "elev": elev[::step].tolist()},
    }


def fmt_hours(h):
    return f"{int(h)} h {int(round((h % 1) * 60)):02d} min"


def _risk(a):
    return a["ai"]["sand"] + a["ai"]["bigrock"]


def _ai_html(ai):
    if not ai:
        return ("<li><b>AI sand/rock risk: unknown here</b> - no analysed AI4Mars photo lies within "
                f"{AI_RADIUS_M:.0f} m of this route.</li>")
    sand = np.array([a["ai"]["sand"] for a in ai])
    rock = np.array([a["ai"]["bigrock"] for a in ai])
    stops = {(a["site"], a["drive"]) for a in ai}
    rows = "".join(
        f"<li>{a['distance_m']:.0f} m away - sol {a['sol_min']}, site {a['site']}/drive {a['drive']} "
        f"({a['loc']}): AI sand <b>{a['ai']['sand']:.0%}</b>, big rock <b>{a['ai']['bigrock']:.0%}</b>"
        + (f"; ground truth sand {a['expert']['sand']:.0%}, big rock {a['expert']['bigrock']:.0%}"
           if a.get("expert") else "")
        + f" - {a['verdict']}</li>" for a in ai[:AI_SHOW])
    worst = max(ai, key=_risk)
    return (f"<li><b>AI sand/rock risk:</b> {len(ai)} analysed photos at {len(stops)} rover stops within "
            f"{AI_RADIUS_M:.0f} m. Near-field sand mean <b>{sand.mean():.0%}</b> (max {sand.max():.0%}), "
            f"big rock mean <b>{rock.mean():.0%}</b> (max {rock.max():.0%}). Riskiest photo: sol "
            f"{worst['sol_min']}, {_risk(worst):.0%} sand + rock.<br>Nearest photos:"
            f"<ul class='mw-dim'>{rows}</ul></li>")


def report_html(r):
    if "error" in r:
        return f"<div class='mw-err'>{r['error']}</div>"
    col = {"LOW": "#1f9d55", "MEDIUM": "#c27c0e", "HIGH": "#c53030"}[r["slope_risk"]]
    ai_sand = (f"{np.mean([a['ai']['sand'] for a in r['ai']]):.0%} / "
               f"{np.mean([a['ai']['bigrock'] for a in r['ai']]):.0%}" if r["ai"] else "n/a")
    legs = "".join(f"<tr><td>{g['leg']}</td><td>{g['km']:.2f} km</td><td>{g['source']}"
                   + (f" ({g['hirise_frac']:.0%} HiRISE)" if g["source"] == "mixed" else "")
                   + f"</td><td>{g['max_slope']:.0f}°</td></tr>" for g in r["legs"])
    wp = " → ".join(f"({a:.4f}, {b:.4f})" for a, b in r["waypoints"])
    stops = "".join(
        f"<li><b>{s['name']}</b> - {s['distance_m'] / 1000:.2f} km from route "
        f"<span class='mw-dim'>({s['kind']}; {s['source']})</span></li>" for s in r["stops"]
    ) or "<li>None within 1.5 km.</li>"
    km, el = np.array(r["profile"]["km"]) / 1000, np.array(r["profile"]["elev"])
    W, H = 320, 70
    svg = ""
    if len(km) > 1 and km[-1] > 0:
        xs = 4 + (W - 8) * km / km[-1]
        ys = H - 6 - (H - 12) * (el - el.min()) / max(el.max() - el.min(), 1)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        svg = (f"<svg viewBox='0 0 {W} {H}' width='100%' height='{H}' role='img' aria-label='elevation profile'>"
               f"<polyline points='{pts}' fill='none' stroke='#2b6cb0' stroke-width='2'/>"
               f"<text x='{W - 4}' y='12' font-size='10' text-anchor='end' fill='currentColor'>{el.max():.0f} m</text>"
               f"<text x='{W - 4}' y='{H - 2}' font-size='10' text-anchor='end' fill='currentColor'>{el.min():.0f} m</text></svg>")
    return f"""
<div class='mw-report'>
  <div class='mw-kpis'>
    <div><span>Distance</span><b>{r['distance_km']:.2f} km</b></div>
    <div><span>Walk time (est.)</span><b>{fmt_hours(r['total_hours'])}</b></div>
    <div><span>Slope risk</span><b style='color:{col}'>{r['slope_risk']}</b></div>
    <div><span>Route on &gt;15° / &gt;25°</span><b>{r['frac_gt15']:.0%} / {r['frac_gt25']:.0%}</b></div>
    <div><span>AI sand / big rock (200 m)</span><b>{ai_sand}</b></div>
    <div><span>On Curiosity's path</span><b>{r['proven_frac']:.0%}</b></div>
    <div><span>Climb / descent</span><b>+{r['climb_m']:.0f} / -{r['descent_m']:.0f} m</b></div>
    <div><span>Max slope</span><b>{r['max_slope_deg']:.0f}°</b></div>
  </div>
  {svg}
  <ul>
    <li>Slope: {r['frac_gt15']:.0%} of the route crosses &gt;15° cells and {r['frac_gt25']:.0%} crosses &gt;25°
      cells (median {r['median_slope_deg']:.0f}°). Sources: <b>HiRISE 1 m DEM for {r['hirise_km']:.2f} km</b>
      (4 m slope baseline, steepest patch per 20 m cell), <b>MOLA 463 m for {r['mola_km']:.2f} km</b>
      (smooths slopes - hazards can be missed).
      <table class='mw-legs'><tr><th>Leg</th><th>Length</th><th>Slope source</th><th>Max</th></tr>{legs}</table></li>
    {_ai_html(r['ai'])}
    <li>{r['proven_frac']:.0%} of this route lies within 100 m of ground Curiosity already drove (proven trafficable).</li>
    <li>Walk time = {fmt_hours(r['walk_hours'])} walking (3 km/h on flat, Tobler slope penalty) +
      {STOP_MIN} min per science stop.</li>
  </ul>
  <div><b>Science stops near the route</b><ul>{stops}</ul></div>
  <div class='mw-dim'>Waypoints (lat, lon): {wp}</div>
</div>"""
