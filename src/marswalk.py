"""Marswalk route analysis on the Gale Crater layers built by src/geo.py.

For a list of waypoints (lat, lon) it returns distance, elevation profile, MOLA slope risk,
how much of the route follows ground Curiosity already drove, science stops nearby and an
estimated walking time.
"""
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.geo import MARS_R, slope_deg  # noqa: E402

GEO = C.ROOT / "geo"
SAMPLE_M = 50.0          # densify the route every 50 m
STOP_RADIUS_M = 1500.0   # science stops "near" the route
PROVEN_M = 100.0         # within 100 m of Curiosity's traverse = rover-proven ground
FLAT_KMH = 3.0           # assumed suited walking pace on flat ground
STOP_MIN = 15            # minutes per science stop


@lru_cache(maxsize=1)
def layers():
    return json.loads((GEO / "layers.json").read_text())


@lru_cache(maxsize=1)
def mola():
    z = np.load(GEO / "mola_dem.npz")
    dem = z["dem"].astype(np.float32)
    s, n, w, e = z["bounds"].tolist()
    ppd = dem.shape[0] / (n - s)
    return dem, slope_deg(dem, s, n, ppd, ppd), (s, n, w, e), ppd


@lru_cache(maxsize=1)
def traverse_xy():
    t = np.array(layers()["traverse"], np.float64)
    return t


def dist_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * MARS_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def densify(points):
    out = [points[0]]
    for (a, b), (c, d) in zip(points[:-1], points[1:]):
        n = max(1, int(math.ceil(dist_m(a, b, c, d) / SAMPLE_M)))
        for k in range(1, n + 1):
            out.append((a + (c - a) * k / n, b + (d - b) * k / n))
    return np.array(out)


def sample_grid(grid, bounds, ppd, lat, lon):
    s, n, w, e = bounds
    r = np.clip(((n - lat) * ppd).astype(int), 0, grid.shape[0] - 1)
    c = np.clip(((lon - w) * ppd).astype(int), 0, grid.shape[1] - 1)
    inside = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
    return grid[r, c], inside


def tobler_kmh(grade):
    """Tobler's hiking function, rescaled so flat ground = FLAT_KMH."""
    return FLAT_KMH * np.exp(-3.5 * np.abs(grade + 0.05)) / math.exp(-3.5 * 0.05)


def analyse(points):
    """points: list of [lat, lon] (>= 2). Returns a dict of route facts."""
    pts = [(float(a), float(b)) for a, b in points]
    if len(pts) < 2:
        return {"error": "Click at least two waypoints on the map."}
    dem, slope, bounds, ppd = mola()
    P = densify(pts)
    lat, lon = P[:, 0], P[:, 1]
    seg = dist_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    total = float(seg.sum())
    elev, inside = sample_grid(dem, bounds, ppd, lat, lon)
    slp, _ = sample_grid(slope, bounds, ppd, lat, lon)
    if not inside.all():
        return {"error": "Part of the route is outside the Gale Crater data window."}

    grade = np.diff(elev) / np.maximum(seg, 1)
    mid_slope = 0.5 * (slp[:-1] + slp[1:])
    hours_walk = float(np.sum(seg / 1000 / tobler_kmh(grade)))

    # rover-proven ground: distance from each route sample to Curiosity's traverse
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

    gt15 = float(np.sum(seg[mid_slope > 15]) / total)
    gt25 = float(np.sum(seg[mid_slope > 25]) / total)
    max_s = float(slp.max())
    level = "HIGH" if max_s > 25 else "MEDIUM" if max_s > 15 else "LOW"
    return {
        "distance_km": total / 1000,
        "elev_start": float(elev[0]), "elev_end": float(elev[-1]),
        "climb_m": float(np.clip(np.diff(elev), 0, None).sum()),
        "descent_m": float(np.clip(-np.diff(elev), 0, None).sum()),
        "max_slope_deg": max_s, "mean_slope_deg": float(slp.mean()),
        "frac_gt15": gt15, "frac_gt25": gt25, "slope_risk": level,
        "proven_frac": proven,
        "stops": stops,
        "walk_hours": hours_walk,
        "total_hours": hours_walk + STOP_MIN * len(science) / 60,
        "profile": {"km": np.concatenate([[0], np.cumsum(seg)]).tolist()[:: max(1, len(P) // 120)],
                    "elev": elev.tolist()[:: max(1, len(P) // 120)]},
    }


def fmt_hours(h):
    return f"{int(h)} h {int(round((h % 1) * 60)):02d} min"


def report_html(r):
    if "error" in r:
        return f"<div class='mw-err'>{r['error']}</div>"
    col = {"LOW": "#1f9d55", "MEDIUM": "#c27c0e", "HIGH": "#c53030"}[r["slope_risk"]]
    stops = "".join(
        f"<li><b>{s['name']}</b> - {s['distance_m'] / 1000:.2f} km from route "
        f"<span class='mw-dim'>({s['kind']}; {s['source']})</span></li>" for s in r["stops"]
    ) or "<li>None within 1.5 km.</li>"
    prof = r["profile"]
    km, el = np.array(prof["km"]) / 1000, np.array(prof["elev"])
    W, H = 320, 70
    if len(km) > 1 and km[-1] > 0:
        xs = 4 + (W - 8) * km / km[-1]
        span = max(el.max() - el.min(), 1)
        ys = H - 6 - (H - 12) * (el - el.min()) / span
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        svg = (f"<svg viewBox='0 0 {W} {H}' width='100%' height='{H}' role='img' "
               f"aria-label='elevation profile'><polyline points='{pts}' fill='none' stroke='#2b6cb0' "
               f"stroke-width='2'/><text x='4' y='12' font-size='10' fill='currentColor'>{el.max():.0f} m</text>"
               f"<text x='4' y='{H - 2}' font-size='10' fill='currentColor'>{el.min():.0f} m</text></svg>")
    else:
        svg = ""
    return f"""
<div class='mw-report'>
  <div class='mw-kpis'>
    <div><span>Distance</span><b>{r['distance_km']:.2f} km</b></div>
    <div><span>Walk time (est.)</span><b>{fmt_hours(r['total_hours'])}</b></div>
    <div><span>Slope risk</span><b style='color:{col}'>{r['slope_risk']}</b></div>
    <div><span>Max slope (MOLA)</span><b>{r['max_slope_deg']:.1f}°</b></div>
    <div><span>Climb / descent</span><b>+{r['climb_m']:.0f} / -{r['descent_m']:.0f} m</b></div>
    <div><span>On Curiosity's path</span><b>{r['proven_frac']:.0%}</b></div>
  </div>
  {svg}
  <ul>
    <li>Slope: {r['frac_gt15']:.0%} of the route crosses &gt;15° cells, {r['frac_gt25']:.0%} crosses &gt;25° cells
      (mean {r['mean_slope_deg']:.1f}°).</li>
    <li>Sand risk: <b>no orbital AI layer</b> - the AI ground check works on rover camera images, and the
      AI4Mars images carry no location. {r['proven_frac']:.0%} of this route lies within 100 m of ground
      Curiosity already drove (proven trafficable).</li>
    <li>Walk time = {fmt_hours(r['walk_hours'])} walking (3 km/h on flat, Tobler slope penalty) +
      {STOP_MIN} min per science stop.</li>
  </ul>
  <div><b>Science stops near the route</b><ul>{stops}</ul></div>
</div>"""
