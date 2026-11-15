"""Build the Gale Crater map layers from official NASA sources into geo/ (small, committed files).

Sources (all verified live):
  HiRISE Gale DEM (1 m)   NASA Mars Trek   Gale_DEM_SMG_1m ImageServer (slope at a 4 m baseline)
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
import urllib.error
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


def get(url, rng=None, tries=5):
    """HTTP GET with retries (the Trek image server returns occasional 500/502s)."""
    import time

    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180) as r:
                data = r.read()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
            if k == tries - 1:
                raise
            print(f"    retry {k + 1} after: {ex}", flush=True)
            time.sleep(3 * (k + 1))
    if rng:
        assert len(data) == rng[1] - rng[0] + 1, f"range read failed: got {len(data)} bytes"
    return data


def png_data_url(rgba):
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


# ---------------- HiRISE 1 m DEM -> slope ----------------
HIRISE_URL = "https://trek.nasa.gov/mars/trekarcgis/rest/services/Gale_DEM_SMG_1m/ImageServer/exportImage"
HI_W, HI_E = 137.12247113204074, 137.72981290693286   # service extent (deg, Mars 2000 sphere)
HI_S, HI_N = -4.925474311955841, -4.248963057145466
HI_RES_M = 4.0        # slope baseline (m): sampled from the 1 m DEM with bilinear interpolation
HI_POOL = 5           # store max slope / mean elevation per 5x5 block = 20 m cells
HI_TILE = (2000, 2000)  # rows, cols per request (multiples of HI_POOL); larger tiles hit server errors
M_PER_DEG = MARS_R * math.pi / 180


def hirise_grid():
    lat0 = 0.5 * (HI_S + HI_N)
    dlat = HI_RES_M / M_PER_DEG
    dlon = HI_RES_M / (M_PER_DEG * math.cos(math.radians(lat0)))
    nr = int((HI_N - HI_S) / dlat) // HI_POOL * HI_POOL
    nc = int((HI_E - HI_W) / dlon) // HI_POOL * HI_POOL
    return dlat, dlon, nr, nc


def fetch_hirise_tile(r0, c0, h, w, dlat, dlon):
    """DEM tile with a 1-pixel halo, cached under DATA_DIR/hirise_tiles."""
    cache = C.DATA_DIR / "hirise_tiles" / f"r{r0}_c{c0}_{h}x{w}.npy"
    if cache.exists():
        return np.load(cache)
    west, east = HI_W + (c0 - 1) * dlon, HI_W + (c0 + w + 1) * dlon
    north, south = HI_N - (r0 - 1) * dlat, HI_N - (r0 + h + 1) * dlat
    url = (f"{HIRISE_URL}?bbox={west},{south},{east},{north}&bboxSR=104905&imageSR=104905"
           f"&size={w + 2},{h + 2}&format=tiff&pixelType=F32&interpolation=RSP_BilinearInterpolation&f=image")
    a = np.array(Image.open(io.BytesIO(get(url))), dtype=np.float32)
    assert a.shape == (h + 2, w + 2), a.shape
    a[(a < -1e4) | (a > 1e4)] = np.nan
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, a)
    return a


def build_hirise():
    """Slope at a 4 m baseline from the HiRISE DEM, kept as max slope per 20 m cell."""
    dlat, dlon, nr, nc = hirise_grid()
    th, tw = HI_TILE
    slope20 = np.zeros((nr // HI_POOL, nc // HI_POOL), np.float32)
    elev20 = np.zeros_like(slope20)
    tiles = [(r0, c0) for r0 in range(0, nr, th) for c0 in range(0, nc, tw)]
    for i, (r0, c0) in enumerate(tiles, 1):
        h, w = min(th, nr - r0), min(tw, nc - c0)
        a = fetch_hirise_tile(r0, c0, h, w, dlat, dlon)
        lats = HI_N - (r0 - 1 + np.arange(h + 2) + 0.5) * dlat
        dx = (dlon * M_PER_DEG * np.cos(np.radians(lats)))[:, None]
        gy, gx = np.gradient(a)
        s = np.degrees(np.arctan(np.hypot(gx / dx, gy / HI_RES_M)))[1:-1, 1:-1]
        z = a[1:-1, 1:-1]
        p = HI_POOL
        blk = lambda x: x.reshape(h // p, p, w // p, p)  # noqa: E731
        slope20[r0 // p:(r0 + h) // p, c0 // p:(c0 + w) // p] = np.nan_to_num(np.nanmax(blk(s), axis=(1, 3)))
        elev20[r0 // p:(r0 + h) // p, c0 // p:(c0 + w) // p] = np.nanmean(blk(z), axis=(1, 3))
        print(f"  HiRISE tile {i}/{len(tiles)}  elev {np.nanmin(z):.0f}..{np.nanmax(z):.0f} m  "
              f"max slope {np.nanmax(s):.1f} deg", flush=True)
    cell = HI_RES_M * HI_POOL
    np.savez_compressed(GEO / "hirise_dem20.npz", slope=np.clip(np.round(slope20), 0, 90).astype(np.uint8),
                        elev=np.round(elev20).astype(np.int16),
                        bounds=np.array([HI_N - nr * dlat, HI_N, HI_W, HI_W + nc * dlon]), cell_m=cell)
    return slope20, (HI_N - nr * dlat, HI_N, HI_W, HI_W + nc * dlon)


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

    slope, (bs, bn, bw, be) = build_hirise()
    print(f"HiRISE slope (max per 20 m cell, 4 m baseline): >15 deg {100 * (slope > 15).mean():.2f}%, "
          f">25 deg {100 * (slope > 25).mean():.2f}% of cells")
    layers["overlays"]["slope"] = {"bounds": [[bs, bw], [bn, be]], "url": png_data_url(slope_overlay(slope)),
                                   "source": "MRO HiRISE Gale DEM (1 m), NASA Mars Trek",
                                   "stats": {"gt15": float((slope > 15).mean()), "gt25": float((slope > 25).mean())}}

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
