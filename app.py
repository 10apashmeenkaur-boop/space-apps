"""Marswalk Planner for Gale Crater, with an AI ground-check layer.

Tab 1  Marswalk Map: NASA Mars Trek CTX + MOLA basemaps, MOLA slope hazards, CRISM mineral
       strength, Curiosity's traverse and science stops (NASA MMGIS). Click waypoints to get slope
       risk, science stops near the route and an estimated walking time.
Tab 2  AI Ground Check: rover Navcam image -> soil / bedrock / sand / big rock map -> safest
       near-field route + briefing, with accuracy on the official AI4Mars test set.
"""
import hashlib
import json
import threading
import time
from collections import OrderedDict

import cv2
import gradio as gr
import numpy as np
from PIL import Image

import config as C
from src.briefing import make_briefing
from src.infer import upsample
from src.marswalk import ai_images, analyse, layers, report_html
from src.planner import near_field_top, plan

VIEW = 640
PAL = np.array([C.COLORS[c] for c in C.CLASSES], dtype=np.float32)
LABELS = {"soil": "Soil", "bedrock": "Bedrock", "sand": "Sand (hazard)", "bigrock": "Big rock (hazard)"}
VERDICT_COLOR = {"GO": "#1f9d55", "GO WITH CAUTION": "#c27c0e", "HOLD": "#c53030"}
METRICS_P = C.RESULTS_DIR / "metrics.json"
METRICS = json.loads(METRICS_P.read_text()) if METRICS_P.exists() else None
SAMPLES = sorted(str(p) for p in C.SAMPLES_DIR.glob("*.png"))
_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


# =====================================================================================
# AI ground check
# =====================================================================================
_MODEL_LOCK = threading.Lock()
_MODELS = {}


def models():
    """DINOv2 + classifier, loaded once (warmed in a background thread at startup)."""
    with _MODEL_LOCK:
        if "m" not in _MODELS:
            from src.features import DinoFeatures
            from src.infer import load_classifier

            _MODELS["m"] = (DinoFeatures(), load_classifier())
        return _MODELS["m"]


threading.Thread(target=models, daemon=True).start()


def to_grey(img: Image.Image):
    notes = []
    if img.mode not in ("L", "I;16", "I", "P", "1", "LA"):
        arr = np.asarray(img.convert("RGB"), dtype=np.int16)
        if np.abs(arr[..., 0] - arr[..., 1]).mean() > 3 or np.abs(arr[..., 1] - arr[..., 2]).mean() > 3:
            notes.append("Colour image converted to grey.")
    grey = np.asarray(img.convert("L"), dtype=np.uint8)
    h, w = grey.shape
    if min(h, w) < 64:
        raise gr.Error("Image is too small (need at least 64x64 pixels).")
    if abs(h - w) / max(h, w) > 0.05:
        notes.append(f"Image is {w}x{h}; it was stretched to a square like Navcam frames.")
    if notes or (h, w) != (C.ORIG, C.ORIG):
        notes.append("The model was trained on Curiosity Navcam images; other cameras may be less accurate.")
    return grey, notes


def predict(grey):
    from src.infer import proba_grid

    key = hashlib.md5(grey.tobytes()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key], 0.0
    dino, clf = models()
    t = time.time()
    prob = proba_grid(clf, dino([grey])[0])
    _CACHE[key] = prob
    if len(_CACHE) > 32:
        _CACHE.popitem(last=False)
    return prob, time.time() - t


def render_overlay(grey, prob, result, alpha=0.5, view=VIEW):
    g = cv2.resize(grey, (view, view), interpolation=cv2.INTER_AREA).astype(np.float32)
    lab = upsample(prob, view).argmax(-1)
    out = ((1 - alpha) * np.repeat(g[..., None], 3, -1) + alpha * PAL[lab]).astype(np.uint8)
    n = result["prob"].shape[0]
    s = view / n
    top = near_field_top(n)
    y = int(top * s)
    for x0 in range(0, view, 24):
        cv2.line(out, (x0, y), (min(x0 + 12, view - 1), y), (255, 255, 255), 2)
    cv2.putText(out, "near-field limit (route zone below)", (8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    for r, c in np.argwhere(result["blocked"][top:]):
        r += top
        cv2.rectangle(out, (int(c * s), int(r * s)), (int((c + 1) * s) - 1, int((r + 1) * s) - 1), (110, 0, 0), 1)

    def pt(rc):
        return int((rc[1] + 0.5) * s), int((rc[0] + 0.5) * s)

    if result["path"]:
        pts = np.array([pt(p) for p in result["path"]], np.int32)
        cv2.polylines(out, [pts], False, (0, 0, 0), 9, cv2.LINE_AA)
        cv2.polylines(out, [pts], False, (40, 230, 90), 5, cv2.LINE_AA)
    cv2.circle(out, pt(result["start"]), 9, (40, 230, 90), -1, cv2.LINE_AA)
    cv2.circle(out, pt(result["goal"]), 10, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(out, pt(result["goal"]), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    return out


def legend_html():
    chip = ("<span style='display:inline-flex;align-items:center;margin-right:14px;white-space:nowrap'>"
            "<span style='width:{w}px;height:{h}px;border-radius:3px;background:rgb{rgb};"
            "display:inline-block;margin-right:6px;border:1px solid #0002'></span>{name}</span>")
    html = "".join(chip.format(w=14, h=14, rgb=C.COLORS[c], name=LABELS[c]) for c in C.CLASSES)
    html += chip.format(w=22, h=5, rgb=(40, 230, 90), name="Safe route")
    html += chip.format(w=12, h=12, rgb=(110, 0, 0), name="Blocked cell")
    return f"<div style='font-size:0.9rem;line-height:2'>{html}</div>"


def briefing_md(brief, secs):
    color = VERDICT_COLOR[brief["verdict"]]
    risk = "—" if brief["risk"] is None else f"{brief['risk']:.2f}"
    head = (f"<div style='display:flex;gap:28px;align-items:baseline;flex-wrap:wrap'>"
            f"<span style='font-size:1.6rem;font-weight:700;color:{color}'>{brief['verdict']}</span>"
            f"<span>Risk score <b>{risk}</b></span>"
            f"<span>Sand in near field <b>{brief['near_field_fractions']['sand']:.0%}</b></span>"
            f"<span>Model time <b>{secs:.1f} s</b></span></div>")
    body = "\n".join(f"- {line}" for line in brief["lines"])
    foot = ("\n\n<small>Risk score = mean P(sand) + P(big rock) along the route (0 = clean, 1 = all hazard). "
            "GO &lt; 0.15 ≤ CAUTION &lt; 0.35 ≤ HOLD.</small>")
    return head + "\n\n" + body + foot


TABLE_HEADERS = ["Test split", "Images", "mIoU (baseline)", "Sand recall (baseline)", "Sand IoU",
                 "Big-rock recall / precision", "Pixel acc (baseline)"]


def metrics_table():
    if METRICS is None:
        return [["metrics.json missing - run src/evaluate.py"] + [""] * 6]
    rows = []
    for s in ["test_min3", "test_min2", "test_min1"]:  # strictest-agreement split first
        r = METRICS["splits"][s]
        m, b = r["model"], r["baseline"]
        mc, bc = m["per_class"], b["per_class"]
        rows.append([s, r["n_images"], f"{m['mIoU']:.3f} ({b['mIoU']:.3f})",
                     f"{mc['sand']['recall']:.1%} ({bc['sand']['recall']:.0%})", f"{mc['sand']['IoU']:.3f}",
                     f"{mc['bigrock']['recall']:.0%} / {mc['bigrock']['precision']:.2f}",
                     f"{m['pixel_accuracy']:.1%} ({b['pixel_accuracy']:.1%})"])
    return rows


def headline():
    if METRICS is None:
        return ""
    r = METRICS["splits"]["test_min3"]
    m, b = r["model"], r["baseline"]
    return (f"**On `test_min3`: mIoU {m['mIoU']:.3f} vs {b['mIoU']:.3f} for the baseline; sand recall "
            f"{m['per_class']['sand']['recall']:.1%} vs {b['per_class']['sand']['recall']:.0%}.** ")


def ground_check(img, auto, tx, ty):
    if img is None:
        return None, "Upload an image or click a sample below."
    grey, notes = to_grey(img)
    for n in notes:
        gr.Warning(n)
    prob, secs = predict(grey)
    n = C.PLAN_GRID
    top = near_field_top(n)
    goal = None if auto else (int(round((n - 1) - ty / 100 * (n - 1 - top))), int(round(tx / 100 * (n - 1))))
    result = plan(prob, goal)
    return render_overlay(grey, prob, result), briefing_md(make_briefing(result), secs)


VERDICT_RANK = {"GO": 0, "GO WITH CAUTION": 1, "HOLD": 2}


def ai_stops():
    """Geolocated AI4Mars images grouped per rover stop (site/drive) for map markers."""
    groups = {}
    for im in ai_images():
        groups.setdefault((im["site"], im["drive"]), []).append(im)
    out = []
    for (site, drive), ims in groups.items():
        ex = [i["expert"] for i in ims if i.get("expert")]
        out.append({"site": site, "drive": drive, "lat": ims[0]["lat"], "lon": ims[0]["lon"],
                    "loc": ims[0]["loc"], "sol": ims[0]["sol_min"], "sol_max": ims[0]["sol_max"],
                    "n": len(ims), "ids": [i["id"] for i in ims],
                    "sand": float(np.mean([i["ai"]["sand"] for i in ims])),
                    "rock": float(np.mean([i["ai"]["bigrock"] for i in ims])),
                    "expert_sand": float(np.mean([e["sand"] for e in ex])) if ex else None,
                    "expert_rock": float(np.mean([e["bigrock"] for e in ex])) if ex else None,
                    "verdict": max((i["verdict"] for i in ims), key=VERDICT_RANK.get)})
    return out


# =====================================================================================
# Marswalk map (Leaflet inside gr.HTML; Python is called through server functions)
# =====================================================================================
def _js_args(data):
    """Gradio passes the JS call's argument list as one value."""
    return list(data) if isinstance(data, (list, tuple)) else ([] if data is None else [data])


def get_layers(data=None):
    L = layers()
    wps = L["waypoints"]
    step = max(1, len(wps) // 120)
    return {**L, "waypoints": wps[::step] + [wps[-1]], "ai_stops": ai_stops()}


def ai_stop_html(data):
    """Popup body for one rover stop: photo | AI terrain map | expert ground truth, with fractions."""
    import base64

    args = _js_args(data)
    ids = args[0] if args and isinstance(args[0], (list, tuple)) else args
    by_id = {i["id"]: i for i in ai_images()}
    parts = []
    for iid in ids[:6]:
        im = by_id.get(iid)
        f = C.ROOT / "geo" / "ai_thumbs" / f"{iid}.jpg"
        if not im or not f.exists():
            continue
        b64 = base64.b64encode(f.read_bytes()).decode()
        e = im.get("expert") or {}
        gt = (f"ground truth sand <b>{e['sand']:.0%}</b>, big rock <b>{e['bigrock']:.0%}</b>" if e
              else "no ground truth in the near field")
        parts.append(
            f"<div style='margin:8px 0'><img src='data:image/jpeg;base64,{b64}' style='width:100%;border-radius:4px'>"
            "<div style='display:flex;justify-content:space-around;font-size:10px;opacity:.75'>"
            "<span>photo</span><span>AI prediction</span><span>expert ground truth</span></div>"
            f"<div style='font-size:11px'><code>{im['file']}</code><br>near field: AI sand <b>{im['ai']['sand']:.0%}</b>, "
            f"big rock <b>{im['ai']['bigrock']:.0%}</b> · {gt} · "
            f"<b style='color:{VERDICT_COLOR[im['verdict']]}'>{im['verdict']}</b></div></div>")
    more = f"<div style='font-size:11px'>+ {len(ids) - 6} more images at this stop</div>" if len(ids) > 6 else ""
    return "".join(parts) + more


def analyse_route(data):
    args = _js_args(data)
    points = args[0] if args and isinstance(args[0], (list, tuple)) and args[0] and \
        isinstance(args[0][0], (list, tuple)) else args
    try:
        return report_html(analyse(points))
    except Exception as ex:  # noqa: BLE001
        return f"<div class='mw-err'>Route analysis failed: {ex}</div>"


LEAFLET_HEAD = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
"""

MAP_HTML = """
<div class="mw-wrap">
  <div class="mw-map"></div>
  <div class="mw-side">
    <div class="mw-help"><b>Plan a Marswalk.</b> Press <b>Draw route</b>, then click waypoints on the map.
      The report updates after each click. Toggle layers top-right.</div>
    <div class="mw-out"><i>No route yet. Try <b>Example route</b>.</i></div>
  </div>
</div>
"""

MAP_CSS = """
.mw-wrap {display:grid; grid-template-columns: minmax(0,1.7fr) minmax(280px,1fr); gap:14px;}
@media (max-width: 900px) {.mw-wrap {grid-template-columns: 1fr;}}
.mw-map {height: 640px; border-radius: 10px; border: 1px solid #8884; background:#222;}
.mw-side {font-size: 0.92rem; line-height: 1.45; max-height: 640px; overflow:auto;}
.mw-help {padding:10px 12px; border-radius:8px; background: #8881; margin-bottom: 10px;}
.mw-kpis {display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:8px; margin-bottom:8px;}
.mw-kpis div {padding:8px 10px; border-radius:8px; background:#8881; display:flex; flex-direction:column;}
.mw-kpis span {font-size:0.75rem; opacity:0.75;}
.mw-kpis b {font-size:1.05rem;}
.mw-dim {opacity:0.7; font-size:0.82rem;}
.mw-err {color:#c53030; font-weight:600;}
.mw-legs {border-collapse:collapse; margin:6px 0; font-size:0.82rem;}
.mw-legs td, .mw-legs th {padding:2px 8px; border-bottom:1px solid #8883; text-align:left;}
.mw-ctl a {display:block; width:auto !important; height:auto !important; line-height:1.3 !important;
  padding:6px 10px; background:#fff; color:#222; font:600 12px system-ui; white-space:nowrap; text-align:left;
  border-bottom:1px solid #ccc; cursor:pointer; text-decoration:none;}
.mw-ctl a.on {background:#2f855a; color:#fff;}
.mw-legend {background:#fffd; color:#222; padding:6px 9px; border-radius:6px; font:12px system-ui; line-height:1.6;}
.mw-legend i {display:inline-block; width:12px; height:12px; margin-right:6px; vertical-align:-2px; border-radius:2px;}
"""

MAP_JS = r"""
const start = async () => {
  if (!window.L) { setTimeout(start, 150); return; }
  const L = window.L;
  const div = element.querySelector('.mw-map');
  if (div._leaflet_id) return;
  const out = element.querySelector('.mw-out');
  const map = L.map(div, {crs: L.CRS.EPSG4326, center: [-4.70, 137.40], zoom: 10, minZoom: 4, maxZoom: 14});

  // --- NASA Mars Trek basemaps (equirectangular, 2x1 tiles at level 0) ---
  const ctx = L.tileLayer('https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX/MapServer/tile/{z}/{y}/{x}', {
    tileSize: 512, zoomOffset: -1, minNativeZoom: 1, maxNativeZoom: 12, maxZoom: 14,
    attribution: 'CTX global mosaic: NASA/JPL/MSSS, Caltech Murray Lab, via NASA Mars Trek'}).addTo(map);
  const molaUrl = 'https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg';
  const mola = L.tileLayer(molaUrl, {maxNativeZoom: 7, maxZoom: 14,
    attribution: 'MOLA colour hillshade: NASA/JPL/GSFC, via NASA Mars Trek'});
  const molaOver = L.tileLayer(molaUrl, {maxNativeZoom: 7, maxZoom: 14, opacity: 0.45});

  const data = await server.get_layers();
  const ov = data.overlays;
  const overlays = {'MOLA hillshade/elevation (tint)': molaOver};
  if (ov.slope) {
    overlays['Slope hazard (HiRISE 1 m DEM): orange >15°, red >25°'] =
      L.imageOverlay(ov.slope.url, ov.slope.bounds, {opacity: 0.8}).addTo(map);
    overlays['HiRISE DEM coverage'] = L.rectangle(ov.slope.bounds,
      {color: '#ffb347', weight: 1, fill: false, dashArray: '6 4'}).addTo(map);
  }
  if (ov.slope_mola) overlays['Slope hazard (MOLA 463 m, outside the HiRISE box)'] =
      L.imageOverlay(ov.slope_mola.url, ov.slope_mola.bounds, {opacity: 0.7}).addTo(map);
  if (ov.crism_sulfate) overlays['CRISM polyhydrated sulfate strength (FRT0000A091)'] =
      L.imageOverlay(ov.crism_sulfate.url, ov.crism_sulfate.bounds).addTo(map);
  if (ov.crism_clay) overlays['CRISM Fe/Mg clay strength (FRT0000A091, weak)'] =
      L.imageOverlay(ov.crism_clay.url, ov.crism_clay.bounds).addTo(map);
  if (ov.crism_sulfate) overlays['CRISM footprint'] = L.rectangle(ov.crism_sulfate.bounds,
      {color: '#ff00c8', weight: 1, fill: false, dashArray: '4 4'}).addTo(map);

  const trav = L.polyline(data.traverse, {color: '#ffffff', weight: 3, opacity: 0.9}).addTo(map);
  overlays["Curiosity traverse (NASA MMGIS)"] = trav;
  const wpl = L.layerGroup(data.waypoints.map(w => L.circleMarker([w.lat, w.lon],
      {radius: 3, color: '#fff', weight: 1, fillColor: '#3182ce', fillOpacity: 1})
      .bindPopup(`<b>Sol ${w.sol}</b> - site ${w.site}, drive ${w.drive}<br>elevation ${w.elev} m<br>odometry ${w.km} km`)));
  overlays['Curiosity waypoints (sol, odometry)'] = wpl;

  const icon = (c) => L.divIcon({className: '', iconSize: [16, 16], iconAnchor: [8, 8],
      html: `<div style="width:14px;height:14px;border-radius:50%;background:${c};border:2px solid #fff;box-shadow:0 0 3px #000"></div>`});
  const kindColor = k => k.startsWith('Orbital') ? '#9f7aea' : k.startsWith('Rover') ? '#e53e3e' : '#ecc94b';
  const sci = L.layerGroup(data.stops.map(s => L.marker([s.lat, s.lon], {icon: icon(kindColor(s.kind))})
      .bindPopup(`<b>${s.name}</b><br>${s.kind}<br><small>${s.source}</small>`))).addTo(map);
  overlays['Science stops'] = sci;

  // --- AI ground checks: AI4Mars Navcam images geolocated by site/drive in their filenames ---
  const vcol = {'GO': '#1f9d55', 'GO WITH CAUTION': '#d69e2e', 'HOLD': '#c53030'};
  const pct = v => v === null ? 'n/a' : Math.round(100 * v) + '%';
  const riskCol = r => r < 0.10 ? '#1a9850' : r < 0.30 ? '#fee08b' : r < 0.50 ? '#fc8d59' : '#d73027';
  const aiLayer = L.layerGroup((data.ai_stops || []).map(s => {
    const risk = s.sand + s.rock;
    const m = L.circleMarker([s.lat, s.lon], {radius: 4 + Math.min(s.n, 8) * 0.6, color: '#111', weight: 1.5,
      fillColor: riskCol(risk), fillOpacity: 0.95});
    const sols = s.sol === s.sol_max ? `sol ${s.sol}` : `sols ${s.sol}-${s.sol_max}`;
    const head = `<b>AI ground check - site ${s.site}, drive ${s.drive}</b> (${sols})<br>` +
      `${s.n} Navcam photo${s.n > 1 ? 's' : ''} · position: ${s.loc}<br>` +
      `predicted near-field sand <b>${pct(s.sand)}</b> + big rock <b>${pct(s.rock)}</b> = <b>${pct(risk)}</b> ` +
      `(ground truth ${pct(s.expert_sand)} + ${pct(s.expert_rock)}) · worst verdict ` +
      `<b style="color:${vcol[s.verdict]}">${s.verdict}</b>`;
    m.bindPopup(head + '<div><i>Loading photos...</i></div>',
                {maxWidth: 480, minWidth: 440, maxHeight: 440, autoPanPaddingTopLeft: [180, 20]});
    m.on('popupopen', async () => {
      if (s.loading || s.html) return;  // fetch once per stop
      s.loading = true;
      try { s.html = await server.ai_stop_html(s.ids); }
      catch (err) { s.html = '<div class="mw-err">Could not load images.</div>'; }
      s.loading = false;
      m.setPopupContent(head + s.html);
    });
    return m;
  })).addTo(map);
  overlays['AI ground checks: predicted sand + big rock % (AI4Mars photos)'] = aiLayer;

  L.control.layers({'CTX mosaic (~6 m/px)': ctx, 'MOLA colour hillshade (463 m/px)': mola}, overlays,
      {collapsed: true}).addTo(map);
  L.control.scale({imperial: false}).addTo(map);

  const legend = L.control({position: 'bottomright'});
  legend.onAdd = () => { const d = L.DomUtil.create('div', 'mw-legend'); d.innerHTML =
    '<i style="background:#ff9600"></i>slope &gt;15° <i style="background:#e61414"></i>&gt;25°<br>' +
    '<b>AI sand + big rock</b><br><i style="background:#1a9850;border-radius:50%"></i>&lt;10% ' +
    '<i style="background:#fee08b;border-radius:50%"></i>10-30% ' +
    '<i style="background:#fc8d59;border-radius:50%"></i>30-50% ' +
    '<i style="background:#d73027;border-radius:50%"></i>&gt;50%<br>' +
    '<i style="background:#ff00c8"></i>sulfate (CRISM)<br><i style="background:#2878ff"></i>clay (CRISM)<br>' +
    '<i style="background:#fff;border:1px solid #333"></i>Curiosity traverse<br>' +
    '<i style="background:#ecc94b;border-radius:50%"></i>science site ' +
    '<i style="background:#9f7aea;border-radius:50%"></i>orbital detection<br>' +
    '<i style="background:#2f855a"></i>your Marswalk'; return d; };
  legend.addTo(map);

  // --- route tool ---
  let drawing = false, pts = [];
  const line = L.polyline([], {color: '#2f855a', weight: 4, dashArray: '8 6'}).addTo(map);
  const marks = L.layerGroup().addTo(map);
  const redraw = async () => {
    line.setLatLngs(pts); marks.clearLayers();
    pts.forEach((p, i) => L.circleMarker(p, {radius: 6, color: '#fff', weight: 2, fillColor: '#2f855a',
      fillOpacity: 1}).bindTooltip(String(i + 1), {permanent: true, direction: 'top', offset: [0, -6]}).addTo(marks));
    if (pts.length < 2) { out.innerHTML = '<i>' + (pts.length ? 'Add another waypoint.' : 'No route yet.') + '</i>'; return; }
    out.innerHTML = '<i>Analysing route...</i>';
    out.innerHTML = await server.analyse_route(pts);
  };
  const ctl = L.control({position: 'topleft'});
  ctl.onAdd = () => {
    const d = L.DomUtil.create('div', 'leaflet-bar mw-ctl');
    d.innerHTML = '<a data-a="draw">✏️ Draw route</a><a data-a="undo">↶ Undo</a><a data-a="clear">✕ Clear</a><a data-a="example">★ Example route</a>';
    L.DomEvent.disableClickPropagation(d);
    d.addEventListener('click', e => {
      const a = e.target.dataset.a; if (!a) return;
      if (a === 'draw') { drawing = !drawing; e.target.classList.toggle('on', drawing);
        div.style.cursor = drawing ? 'crosshair' : ''; }
      if (a === 'undo') { pts.pop(); redraw(); }
      if (a === 'clear') { pts = []; redraw(); }
      if (a === 'example') { pts = [[-4.5895, 137.4416], [-4.6268, 137.4069], [-4.6380, 137.4015], [-4.6634, 137.3776]];
        map.fitBounds(L.latLngBounds(pts).pad(0.4)); redraw(); }
    });
    return d;
  };
  ctl.addTo(map);
  map.on('click', e => { if (!drawing) return; pts.push([+e.latlng.lat.toFixed(6), +e.latlng.lng.toFixed(6)]); redraw(); });
  setTimeout(() => map.invalidateSize(), 300);
};
start();
"""


# =====================================================================================
# Layout
# =====================================================================================
CSS = ".gradio-container {max-width: 1360px !important}"

with gr.Blocks(title="Marswalk Planner - Gale Crater") as demo:
    gr.Markdown(
        "# 🛰️ Marswalk Planner: Gale Crater\n"
        "A layered view of Curiosity's field site from **multiple NASA missions**: MRO CTX imagery, "
        "**MRO HiRISE 1 m slope hazards**, MGS MOLA topography, MRO CRISM / Mars Express OMEGA mineral signals, "
        "Curiosity's traverse and science stops, and **319 AI ground checks of Curiosity Navcam images placed on "
        "the traverse** by the site/drive in their filenames. Plan a walk on the map: the route report "
        "combines slope, AI sand risk, science stops and walking time.")

    with gr.Tab("🗺️ Marswalk Map"):
        gr.HTML(value="", html_template=MAP_HTML, css_template=MAP_CSS, js_on_load=MAP_JS, head=LEAFLET_HEAD,
                server_functions=[get_layers, analyse_route, ai_stop_html], apply_default_css=False)
        gr.Markdown(
            "<small>**AI ground checks** are the 319 Curiosity Navcam images of the official AI4Mars expert test "
            "set that could be placed: each filename (e.g. `NLB_537932733EDR_F0600888NCAM00320M1`) carries the "
            "rover's site (060) and drive (0888), matched to NASA MMGIS waypoints - 287 exact, 32 interpolated "
            "within the site. Markers sit at the rover position; the terrain shown is within ~30 m of it. "
            "Click a marker to see each image, the AI terrain map and the experts' sand fraction.</small>")

    with gr.Tab("🔬 AI Ground Check"):
        gr.Markdown("Upload a rover Navcam image (or click a sample): the model maps **soil, bedrock, sand and "
                    "big rocks**, then previews the safest route through the near field.")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Navcam image (greyscale, ideally 1024×1024)", height=380,
                               sources=["upload", "clipboard"])
                if SAMPLES:
                    gr.Examples(SAMPLES, inputs=inp, label="Samples from the AI4Mars expert test set "
                                "(never used in training)", examples_per_page=8)
                auto = gr.Checkbox(True, label="Auto target: farthest safe point straight ahead")
                tx = gr.Slider(0, 100, 50, step=1, label="Target across (left → right)", interactive=False)
                ty = gr.Slider(0, 100, 100, step=1, label="Target distance (near → far edge of near field)",
                               interactive=False)
                btn = gr.Button("Plan route", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Terrain map + safe route (image-space preview)", height=520,
                                   interactive=False, format="png")
                gr.HTML(legend_html())
                out_md = gr.Markdown("Upload an image or click a sample.")

        with gr.Accordion("How accurate is this?", open=False):
            gr.Markdown(
                headline() +
                "Scored pixel by pixel on the **official AI4Mars expert test set** (322 Curiosity Navcam images, "
                "never seen in training), ignoring unlabelled pixels. "
                f"Classifier: **{METRICS['model'] if METRICS else '?'}** on frozen DINOv2-small patch features. "
                f"Baseline = always predict **{METRICS['baseline_class'] if METRICS else '?'}** "
                "(the most common training class), so its sand recall is 0%. `test_min3` scores only pixels "
                "that all of the (at least three) labellers agreed on; `min2` and `min1` relax that.")
            gr.Dataframe(metrics_table(), headers=TABLE_HEADERS, interactive=False, wrap=True)
            with gr.Row():
                cm_png = C.RESULTS_DIR / "confusion_matrix.png"
                if cm_png.exists():
                    gr.Image(str(cm_png), label="Confusion matrix, test_min3 (rows = expert label)",
                             interactive=False, height=380)
                gr.Markdown(
                    "**Published reference points - a different setup, so not a like-for-like comparison:** the AI4Mars paper (CVPR 2021 workshop) reports **DeepLabv3 above 96% overall accuracy**, and the Zooniverse AI4Mars results report **SPOCv2 at 91.41%**. Those models were trained and scored under their own protocols; ours scores pixels of the public test masks, and `test_min3` keeps only pixels that all of the (at least three) labellers agreed on, which is an easier target. We do **not** claim to beat JPL's models. Ours is a frozen foundation model plus a small classifier that runs on a laptop CPU.\n\n"
                    "**Honest limits**\n"
                    "- Big rocks are very rare (0.9% of training pixels, 0.1–0.2% of test pixels) and the weakest class.\n"
                    "- Labels only reach ~30 m, so the far field is not planned.\n"
                    "- Each prediction cell covers 32×32 pixels, so thin boundaries are blurred.\n"
                    "- The route is drawn on the camera view, not a top-down map: a preview for operators, "
                    "not distances in metres.\n"
                    "- Trained on Curiosity Navcam only; other cameras may be less accurate.")

        ins, outs = [inp, auto, tx, ty], [out_img, out_md]
        auto.change(lambda a: (gr.update(interactive=not a), gr.update(interactive=not a)), auto, [tx, ty])
        inp.change(ground_check, ins, outs)
        auto.change(ground_check, ins, outs)
        tx.release(ground_check, ins, outs)
        ty.release(ground_check, ins, outs)
        btn.click(ground_check, ins, outs)

    with gr.Tab("ℹ️ Data & limits"):
        gr.Markdown(
            "### NASA data used\n"
            "| Layer | Mission / instrument | Source |\n|---|---|---|\n"
            "| CTX global mosaic (~6 m/px) | MRO Context Camera | NASA Mars Trek WMTS (Caltech Murray Lab beta01 mosaic) |\n"
            "| MOLA colour hillshade (463 m/px) | MGS Mars Orbiter Laser Altimeter | NASA Mars Trek WMTS |\n"
            "| Elevation + slope hazards (1 m DEM, 4 m slope baseline, 20 m cells) | MRO HiRISE stereo DEM of Gale | NASA Mars Trek (Gale_DEM_SMG_1m) |\n"
            "| Clay / sulfate strength | MRO CRISM MTRDR FRT0000A091 (D2300, SINDEX2, BD1900_2) | PDS Geosciences Node |\n"
            "| Hydrous mineral detections | MRO CRISM + Mars Express OMEGA | NASA Mars Trek |\n"
            "| Traverse, waypoints, science sites | MSL Curiosity | NASA MMGIS (mars.nasa.gov/maps) |\n"
            "| Terrain labels for the AI | MSL Curiosity Navcam | NASA/JPL-Caltech AI4Mars |\n"
            "| Image positions | site/drive in Navcam product IDs + MSL waypoints | AI4Mars filenames + NASA MMGIS |\n\n"
            "### Limits\n"
            "- **Slope covers the HiRISE DEM area only** (137.12–137.73°E, 4.25–4.93°S, around Curiosity's "
            "traverse). Slope is computed at a 4 m baseline and the steepest patch per 20 m cell is kept, so a "
            "flagged cell may be mostly walkable. Mosaic seams in the DEM can create false steep lines.\n"
            "- **CRISM layers show band-parameter strength from one observation**, not confirmed mineral maps; "
            "the clay signal in FRT0000A091 is weak. Orbital detections are only given to 0.1°.\n"
            "- **AI sand risk exists only where Curiosity took a test-set image**: 319 images at 89 rover "
            "stops along the traverse. Positions are the rover's, interpolated by drive number when the exact "
            "drive is not a published waypoint (32 images); 3 images could not be placed.\n"
            "- **Walk time** assumes 3 km/h on flat ground with Tobler's slope penalty, plus 15 min per science "
            "stop. It is a planning estimate, not a validated EVA model.")

    gr.Markdown("<small>Data: NASA/JPL-Caltech AI4Mars; NASA Mars Trek (CTX, MOLA, CRISM/OMEGA); PDS Geosciences "
                "Node (CRISM MTRDR); MRO HiRISE Gale DEM via NASA Mars Trek; NASA MMGIS (Curiosity traverse). "
                "Model: DINOv2-small (Meta) + LightGBM. Routing: A* (images) and great-circle sampling on the "
                "HiRISE DEM (map).</small>")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(css=CSS, theme=gr.themes.Soft())
