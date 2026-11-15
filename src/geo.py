"""Build the Gale Crater map layers from official NASA sources into geo/ (small, committed files).

Sources (all verified live):
  MOLA MEGDR 128 px/deg   PDS Geosciences  megt00n090hb.img (range-read Gale rows only)
  CRISM MTRDR summary     PDS Geosciences  FRT0000A091_07_SU165J_MTR3 (bands D2300, SINDEX2, BD1900_2)
  CRISM/OMEGA detections  NASA Mars Trek   Hydrous_Mineral_Detections_OMEGA_CRISM_clon0
  Curiosity traverse      NASA MMGIS       MSL_traverse.json, MSL_waypoints.json, MSL_SurfaceSites.json

Run:  python src/geo.py
"""
import base64
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

GEO = C.ROOT / "geo"
MARS_R = 3396190.0  # m, IAU Mars 2000 sphere

# Gale Crater window (deg). Crater centre ~5.4S 137.8E, Curiosity area ~4.6-4.9S 137.4E.
LAT_N, LAT_S, LON_W, LON_E = -3.0, -7.5, 135.5, 140.0

MEGDR = ("https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg128/megt00n090hb.img")
MEGDR_PPD, MEGDR_COLS, MEGDR_TOP, MEGDR_WEST = 128, 11520, 0.0, 90.0

CRISM_ID = "FRT0000A091"
CRISM_URL = ("https://pds-geosciences.wustl.edu/mro/mro-m-crism-5-rdr-mptargeted-v1/mrocr_4001/mtrdr/2008/"
             "2008_051/frt0000a091/frt0000a091_07_su165j_mtr3.img")
CRISM_SHAPE = (712, 764)  # lines, samples (from the PDS label)
CRISM_BOUNDS = (-5.2951956, -5.0792758, 137.3385500, 137.5711400)  # S, N, W, E
CRISM_BANDS = {"BD1900_2": 25, "D2300": 39, "SINDEX2": 41}

TREK_HYDROUS = ("https://trek.nasa.gov/mars/trekarcgis/rest/services/Hydrous_Mineral_Detections_OMEGA_CRISM_clon0/"
                "MapServer/0/query?where=1%3D1&geometry={w},{s},{e},{n}&geometryType=esriGeometryEnvelope"
                "&inSR=104905&spatialRel=esriSpatialRelIntersects&outFields=*&f=geojson")
MMGIS = "https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/"

UA = {"User-Agent": "sandtrap-geo/1.0"}


def get(url, rng=None):
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
        data = r.read()
    if rng:
        assert len(data) == rng[1] - rng[0] + 1, f"range read failed: got {len(data)} bytes"
    return data


def png_data_url(rgba):
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


# ---------------- MOLA elevation + slope ----------------
def build_mola():
    r0 = int(round((MEGDR_TOP - LAT_N) * MEGDR_PPD))
    r1 = int(round((MEGDR_TOP - LAT_S) * MEGDR_PPD))
    c0 = int(round((LON_W - MEGDR_WEST) * MEGDR_PPD))
    c1 = int(round((LON_E - MEGDR_WEST) * MEGDR_PPD))
    row_bytes = MEGDR_COLS * 2
    raw = get(MEGDR, (r0 * row_bytes, r1 * row_bytes - 1))
    dem = np.frombuffer(raw, ">i2").reshape(r1 - r0, MEGDR_COLS)[:, c0:c1].astype(np.int16)
    np.savez_compressed(GEO / "mola_dem.npz", dem=dem, bounds=np.array([LAT_S, LAT_N, LON_W, LON_E]))
    print(f"MOLA DEM {dem.shape}, elevation {dem.min()}..{dem.max()} m")
    return dem


def slope_deg(dem, lat_s, lat_n, ppd_lat, ppd_lon):
    """Slope in degrees for an equirectangular grid (rows north->south)."""
    h, w = dem.shape
    lats = lat_n - (np.arange(h) + 0.5) / ppd_lat
    dy = MARS_R * math.radians(1 / ppd_lat)
    dx = (MARS_R * np.radians(1 / ppd_lon) * np.cos(np.radians(lats)))[:, None]
    gy, gx = np.gradient(dem.astype(np.float64))
    return np.degrees(np.arctan(np.hypot(gx / dx, gy / dy))).astype(np.float32)


def slope_overlay(slope):
    rgba = np.zeros(slope.shape + (4,), np.uint8)
    rgba[slope > 15] = (255, 150, 0, 170)
    rgba[slope > 25] = (230, 20, 20, 210)
    return rgba


# ---------------- CRISM ----------------
def build_crism():
    nl, ns = CRISM_SHAPE
    band_bytes = nl * ns * 4
    bands = {}
    for name, k in CRISM_BANDS.items():
        raw = get(CRISM_URL, (k * band_bytes, (k + 1) * band_bytes - 1))
        b = np.frombuffer(raw, "<f4").reshape(nl, ns).astype(np.float32)
        b[(b > 1e3) | (b < -1e3) | ~np.isfinite(b)] = np.nan  # 65535 = no data
        bands[name] = b
        v = b[np.isfinite(b)]
        print(f"  CRISM {name}: valid {v.size}  p50 {np.percentile(v, 50):.4f}  p99 {np.percentile(v, 99):.4f}")
    np.savez_compressed(C.DATA_DIR / "crism_su.npz", **bands)
    return bands


CRISM_STRETCH = {  # parameter, low, high  (stretched like CRISM browse products)
    "clay": ("D2300", 0.002, 0.008),      # Fe/Mg-OH band depth: smectite-type clays
    "sulfate": ("SINDEX2", 0.012, 0.025),  # polyhydrated-sulfate index (also needs hydration, BD1900_2)
}


def crism_overlays(bands):
    """Strength maps, not confirmed mineral maps: alpha grows with the band parameter."""
    import cv2

    out = {}
    hyd = cv2.medianBlur(np.nan_to_num(bands["BD1900_2"]).astype(np.float32), 5)
    for name, col in [("clay", (40, 120, 255)), ("sulfate", (255, 0, 200))]:
        p, lo, hi = CRISM_STRETCH[name]
        v = cv2.medianBlur(np.nan_to_num(bands[p], nan=lo).astype(np.float32), 5)  # despeckle
        a = np.clip((v - lo) / (hi - lo), 0, 1)
        if name == "sulfate":
            a = a * (hyd > 0.003)
        rgba = np.zeros(v.shape + (4,), np.uint8)
        rgba[..., :3] = col
        rgba[..., 3] = (a * 230).astype(np.uint8)
        out[name] = (rgba, float((a > 0.5).mean()))
    return out


# ---------------- vectors ----------------
def build_vectors():
    trav = json.loads(get(MMGIS + "MSL_traverse.json"))
    line = []
    for f in trav["features"]:
        g = f["geometry"]
        parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for p in parts:
            line.extend([[round(c[1], 6), round(c[0], 6)] for c in p])
    # thin to every 3rd vertex (keeps the shape, ~1/3 size)
    thin = line[::3] + [line[-1]]

    wps = json.loads(get(MMGIS + "MSL_waypoints.json"))["features"]
    waypoints = [{"sol": p["properties"]["sol"], "site": p["properties"]["site"],
                  "drive": p["properties"]["drive"], "lat": round(p["properties"]["lat"], 6),
                  "lon": round(p["properties"]["lon"], 6), "elev": round(p["properties"]["elev_geoid"], 1),
                  "km": p["properties"]["dist_km"]} for p in wps]

    stops = []
    for f in json.loads(get(MMGIS + "MSL_SurfaceSites.json"))["features"]:
        ring = np.array(f["geometry"]["coordinates"][0])
        stops.append({"name": f["properties"]["Name"], "lat": float(ring[:, 1].mean()),
                      "lon": float(ring[:, 0].mean()), "kind": "Curiosity science site",
                      "source": "NASA MMGIS (MSL Surface View)"})
    first, last = waypoints[0], waypoints[-1]
    stops.append({"name": f"Bradbury Landing (sol {first['sol']})", "lat": first["lat"], "lon": first["lon"],
                  "kind": "Curiosity landing site", "source": "NASA MMGIS waypoints"})
    stops.append({"name": f"Curiosity, latest position (sol {last['sol']})", "lat": last["lat"],
                  "lon": last["lon"], "kind": "Rover position", "source": "NASA MMGIS waypoints"})

    hyd = json.loads(get(TREK_HYDROUS.format(w=LON_W, s=LAT_S, e=LON_E, n=LAT_N)))["features"]
    for f in hyd:
        p = f["properties"]
        stops.append({"name": f"Hydrous mineral detection ({p['Instrument']})", "lat": p["Latitude"],
                      "lon": p["Long_East"], "kind": "Orbital mineral detection (0.1° precision)",
                      "source": "NASA Mars Trek: Hydrous Mineral Detections (OMEGA, CRISM)"})
    print(f"traverse {len(line)} vertices (kept {len(thin)}), waypoints {len(waypoints)}, stops {len(stops)}")
    return thin, waypoints, stops


def main():
    GEO.mkdir(exist_ok=True)
    layers = {"window": {"south": LAT_S, "north": LAT_N, "west": LON_W, "east": LON_E}, "overlays": {}}

    dem = build_mola()
    slope = slope_deg(dem, LAT_S, LAT_N, MEGDR_PPD, MEGDR_PPD)
    
    print(f"MOLA slope: max {slope.max():.1f} deg, >15 deg {100 * (slope > 15).mean():.2f}%, "
          f">25 deg {100 * (slope > 25).mean():.3f}%")
    layers["overlays"]["slope"] = {"bounds": [[LAT_S, LON_W], [LAT_N, LON_E]],
                                   "url": png_data_url(slope_overlay(slope)),
                                   "stats": {"max": float(slope.max()), "gt15": float((slope > 15).mean()),
                                             "gt25": float((slope > 25).mean())}}

    try:
        bands = build_crism()
        s, n, w, e = CRISM_BOUNDS
        for name, (rgba, frac) in crism_overlays(bands).items():
            layers["overlays"][f"crism_{name}"] = {"bounds": [[s, w], [n, e]], "url": png_data_url(rgba),
                                                    "stats": {"fraction": frac}, "obs": CRISM_ID}
            print(f"  CRISM {name}: {100 * frac:.1f}% of footprint flagged")
    except Exception as ex:  # noqa: BLE001
        print(f"CRISM layer skipped: {ex}")

    trav, wps, stops = build_vectors()
    layers.update(traverse=trav, waypoints=wps, stops=stops)
    (GEO / "layers.json").write_text(json.dumps(layers))
    print(f"wrote {GEO / 'layers.json'} ({(GEO / 'layers.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
