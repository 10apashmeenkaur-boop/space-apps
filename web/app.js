/* Static Marswalk Planner: loads precomputed data files, then runs the map and route tool in the browser. */
(async function () {
  // ---- tabs ----
  document.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b));
    document.querySelectorAll(".tab").forEach(t => t.classList.toggle("on", t.id === b.dataset.t));
    if (b.dataset.t === "t-map" && window._map) setTimeout(() => window._map.invalidateSize(), 50);
  }));

  const getJSON = u => fetch(u).then(r => { if (!r.ok) throw new Error(u + " " + r.status); return r.json(); });
  const getBin = (u, T) => fetch(u).then(r => { if (!r.ok) throw new Error(u + " " + r.status); return r.arrayBuffer(); }).then(b => new T(b));
  const out = document.getElementById("out");

  let meta, layers, ai, aiStops;
  try {
    [meta, layers, ai, aiStops] = await Promise.all([getJSON("data/grids.json"), getJSON("data/layers.json"),
      getJSON("data/ai_images.json"), getJSON("data/ai_stops.json")]);
    const [hs, he, ms, me] = await Promise.all([getBin("data/hirise_slope.u8", Uint8Array), getBin("data/hirise_elev.i16", Int16Array),
      getBin("data/mola_slope.f32", Float32Array), getBin("data/mola_elev.i16", Int16Array)]);
    MW.init({hirise: {...meta.hirise, slope: hs, elev: he}, mola: {...meta.mola, slope: ms, elev: me}, layers, ai});
  } catch (e) {
    out.innerHTML = `<div class="mw-err">Could not load data: ${e.message}</div>`;
    return;
  }
  out.innerHTML = "<i>No route yet. Try <b>★ Example route</b>.</i>";

  // ---- map ----
  const map = L.map("map", {crs: L.CRS.EPSG4326, center: [-4.70, 137.40], zoom: 10, minZoom: 4, maxZoom: 14});
  window._map = map;
  const ctx = L.tileLayer("https://astro.arcgis.com/arcgis/rest/services/OnMars/CTX/MapServer/tile/{z}/{y}/{x}", {
    tileSize: 512, zoomOffset: -1, minNativeZoom: 1, maxNativeZoom: 12, maxZoom: 14,
    attribution: "CTX global mosaic: NASA/JPL/MSSS/The Murray Lab, via NASA Mars Trek"}).addTo(map);
  const molaUrl = "https://trek.nasa.gov/tiles/Mars/EQ/Mars_MGS_MOLA_ClrShade_merge_global_463m/1.0.0/default/default028mm/{z}/{y}/{x}.jpg";
  const mola = L.tileLayer(molaUrl, {maxNativeZoom: 7, maxZoom: 14, attribution: "MOLA colour hillshade: NASA/JPL/GSFC, via NASA Mars Trek"});
  const ov = layers.overlays, overlays = {"MOLA hillshade/elevation (tint)": L.tileLayer(molaUrl, {maxNativeZoom: 7, maxZoom: 14, opacity: 0.45})};
  if (ov.slope) {
    overlays["Slope hazard (HiRISE 1 m DEM): orange >15°, red >25°"] = L.imageOverlay(ov.slope.url, ov.slope.bounds, {opacity: 0.8}).addTo(map);
    overlays["HiRISE DEM coverage"] = L.rectangle(ov.slope.bounds, {color: "#ffb347", weight: 1, fill: false, dashArray: "6 4"}).addTo(map);
  }
  if (ov.slope_mola) overlays["Slope hazard (MOLA 463 m, outside the HiRISE box)"] = L.imageOverlay(ov.slope_mola.url, ov.slope_mola.bounds, {opacity: 0.7}).addTo(map);
  if (ov.crism_sulfate) overlays["CRISM polyhydrated sulfate strength (FRT0000A091)"] = L.imageOverlay(ov.crism_sulfate.url, ov.crism_sulfate.bounds).addTo(map);
  if (ov.crism_clay) overlays["CRISM Fe/Mg clay strength (FRT0000A091, weak)"] = L.imageOverlay(ov.crism_clay.url, ov.crism_clay.bounds).addTo(map);
  if (ov.crism_sulfate) overlays["CRISM footprint"] = L.rectangle(ov.crism_sulfate.bounds, {color: "#ff00c8", weight: 1, fill: false, dashArray: "4 4"}).addTo(map);
  overlays["Curiosity traverse (NASA MMGIS)"] = L.polyline(layers.traverse, {color: "#ffffff", weight: 3, opacity: 0.9}).addTo(map);
  overlays["Curiosity waypoints (sol, odometry)"] = L.layerGroup(layers.waypoints.map(w => L.circleMarker([w.lat, w.lon],
    {radius: 3, color: "#fff", weight: 1, fillColor: "#3182ce", fillOpacity: 1})
    .bindPopup(`<b>Sol ${w.sol}</b> - site ${w.site}, drive ${w.drive}<br>elevation ${w.elev} m<br>odometry ${w.km} km`)));
  const icon = c => L.divIcon({className: "", iconSize: [16, 16], iconAnchor: [8, 8],
    html: `<div style="width:14px;height:14px;border-radius:50%;background:${c};border:2px solid #fff;box-shadow:0 0 3px #000"></div>`});
  const kindColor = k => k.startsWith("Orbital") ? "#9f7aea" : k.startsWith("Rover") ? "#e53e3e" : "#ecc94b";
  overlays["Science stops"] = L.layerGroup(layers.stops.map(s => L.marker([s.lat, s.lon], {icon: icon(kindColor(s.kind))})
    .bindPopup(`<b>${s.name}</b><br>${s.kind}<br><small>${s.source}</small>`))).addTo(map);

  // AI ground-check markers, coloured by predicted sand + big rock
  const vcol = {"GO": "#1f9d55", "GO WITH CAUTION": "#d69e2e", "HOLD": "#c53030"};
  const pct = v => v === null || v === undefined ? "n/a" : Math.round(100 * v) + "%";
  const riskCol = r => r < 0.10 ? "#1a9850" : r < 0.30 ? "#fee08b" : r < 0.50 ? "#fc8d59" : "#d73027";
  const byId = Object.fromEntries(ai.map(a => [a.id, a]));
  overlays["AI ground checks: predicted sand + big rock % (AI4Mars photos)"] = L.layerGroup(aiStops.map(s => {
    const risk = s.sand + s.rock;
    const m = L.circleMarker([s.lat, s.lon], {radius: 4 + Math.min(s.n, 8) * 0.6, color: "#111", weight: 1.5, fillColor: riskCol(risk), fillOpacity: 0.95});
    const sols = s.sol === s.sol_max ? `sol ${s.sol}` : `sols ${s.sol}-${s.sol_max}`;
    const photos = s.ids.slice(0, 6).map(id => {
      const a = byId[id], e = a.expert;
      const gt = e ? `ground truth sand <b>${pct(e.sand)}</b>, big rock <b>${pct(e.bigrock)}</b>` : "no ground truth in the near field";
      return `<div style="margin:8px 0"><img src="thumbs/${id}.jpg" loading="lazy" style="width:100%;border-radius:4px" alt="Navcam photo, AI prediction and expert labels">
        <div style="display:flex;justify-content:space-around;font-size:10px;opacity:.75"><span>photo</span><span>AI prediction</span><span>expert ground truth</span></div>
        <div style="font-size:11px"><code>${a.file}</code><br>near field: AI sand <b>${pct(a.ai.sand)}</b>, big rock <b>${pct(a.ai.bigrock)}</b> · ${gt} ·
        <b style="color:${vcol[a.verdict]}">${a.verdict}</b></div></div>`;
    }).join("") + (s.ids.length > 6 ? `<div style="font-size:11px">+ ${s.ids.length - 6} more photos at this stop</div>` : "");
    m.bindPopup(`<b>AI ground check - site ${s.site}, drive ${s.drive}</b> (${sols})<br>${s.n} Navcam photo${s.n > 1 ? "s" : ""} · position: ${s.loc}<br>` +
      `predicted near-field sand <b>${pct(s.sand)}</b> + big rock <b>${pct(s.rock)}</b> = <b>${pct(risk)}</b> ` +
      `(ground truth ${pct(s.expert_sand)} + ${pct(s.expert_rock)}) · worst verdict <b style="color:${vcol[s.verdict]}">${s.verdict}</b>` + photos,
      {maxWidth: 400, minWidth: 340, maxHeight: 440, autoPanPaddingTopLeft: [40, 20], autoPanPaddingBottomRight: [20, 20]});
    return m;
  })).addTo(map);

  L.control.layers({"CTX mosaic (~6 m/px)": ctx, "MOLA colour hillshade (463 m/px)": mola}, overlays, {collapsed: true}).addTo(map);
  L.control.scale({imperial: false}).addTo(map);
  const legend = L.control({position: "bottomright"});
  legend.onAdd = () => { const d = L.DomUtil.create("div", "mw-legend"); d.innerHTML =
    '<i style="background:#ff9600"></i>slope &gt;15° <i style="background:#e61414"></i>&gt;25°<br>' +
    '<b>AI sand + big rock</b><br><i style="background:#1a9850;border-radius:50%"></i>&lt;10% ' +
    '<i style="background:#fee08b;border-radius:50%"></i>10-30% <i style="background:#fc8d59;border-radius:50%"></i>30-50% ' +
    '<i style="background:#d73027;border-radius:50%"></i>&gt;50%<br>' +
    '<i style="background:#ff00c8"></i>sulfate (CRISM) <i style="background:#2878ff"></i>clay (CRISM)<br>' +
    '<i style="background:#fff;border:1px solid #333"></i>Curiosity traverse<br>' +
    '<i style="background:#ecc94b;border-radius:50%"></i>science site <i style="background:#9f7aea;border-radius:50%"></i>orbital detection<br>' +
    '<i style="background:#2f855a"></i>your Marswalk'; return d; };
  legend.addTo(map);

  // ---- route tool ----
  let drawing = false, pts = [];
  const line = L.polyline([], {color: "#2f855a", weight: 4, dashArray: "8 6"}).addTo(map);
  const marks = L.layerGroup().addTo(map);
  function redraw() {
    line.setLatLngs(pts); marks.clearLayers();
    pts.forEach((p, i) => L.circleMarker(p, {radius: 6, color: "#fff", weight: 2, fillColor: "#2f855a", fillOpacity: 1})
      .bindTooltip(String(i + 1), {permanent: true, direction: "top", offset: [0, -6]}).addTo(marks));
    if (pts.length < 2) { out.innerHTML = "<i>" + (pts.length ? "Add another waypoint." : "No route yet.") + "</i>"; return; }
    try { out.innerHTML = MW.reportHTML(MW.analyse(pts)); }
    catch (e) { out.innerHTML = `<div class="mw-err">Route analysis failed: ${e.message}</div>`; }
  }
  const ctl = L.control({position: "topleft"});
  ctl.onAdd = () => {
    const d = L.DomUtil.create("div", "leaflet-bar mw-ctl");
    d.innerHTML = '<a data-a="draw">✏️ Draw route</a><a data-a="undo">↶ Undo</a><a data-a="clear">✕ Clear</a><a data-a="example">★ Example route</a>';
    L.DomEvent.disableClickPropagation(d);
    d.addEventListener("click", e => {
      const a = e.target.dataset.a; if (!a) return;
      if (a === "draw") { drawing = !drawing; e.target.classList.toggle("on", drawing); map.getContainer().style.cursor = drawing ? "crosshair" : ""; }
      if (a === "undo") { pts.pop(); redraw(); }
      if (a === "clear") { pts = []; redraw(); }
      if (a === "example") { pts = [[-4.5895, 137.4416], [-4.6268, 137.4069], [-4.6380, 137.4015], [-4.6634, 137.3776]];
        map.fitBounds(L.latLngBounds(pts).pad(0.4)); redraw(); }
    });
    return d;
  };
  ctl.addTo(map);
  map.on("click", e => { if (!drawing) return; pts.push([+e.latlng.lat.toFixed(6), +e.latlng.lng.toFixed(6)]); redraw(); });
  window._mw = {setRoute: p => { pts = p; map.fitBounds(L.latLngBounds(pts).pad(0.25)); redraw(); }};

  // ---- AI tab (precomputed) ----
  try {
    const [samples, metrics] = await Promise.all([getJSON("data/ai_samples.json"), getJSON("data/metrics.json")]);
    document.getElementById("ai-cards").innerHTML = samples.map(s => `<div class="card">
      <img src="${s.overlay}" alt="AI terrain map and safe route for ${s.name}" loading="lazy">
      <h3>${s.name} · <span class="v-${s.verdict.replace(/ /g, "-")}">${s.verdict}</span></h3>${s.briefing}</div>`).join("");
    const rows = ["test_min3", "test_min2", "test_min1"].map(k => { const r = metrics.splits[k], m = r.model, b = r.baseline, pc = m.per_class, bc = b.per_class;
      return `<tr><td>${k}</td><td>${r.n_images}</td><td><b>${m.mIoU.toFixed(3)}</b> (${b.mIoU.toFixed(3)})</td>
        <td><b>${(100 * pc.sand.recall).toFixed(1)}%</b> (${Math.round(100 * bc.sand.recall)}%)</td><td>${pc.sand.IoU.toFixed(3)}</td>
        <td>${Math.round(100 * pc.bigrock.recall)}% / ${pc.bigrock.precision.toFixed(2)}</td>
        <td>${(100 * m.pixel_accuracy).toFixed(1)}% (${(100 * b.pixel_accuracy).toFixed(1)}%)</td></tr>`; }).join("");
    document.getElementById("metrics").innerHTML = `<p class="note">Scored pixel by pixel on the official AI4Mars expert test set
      (322 Curiosity Navcam images, never seen in training). Baseline = always predict "${metrics.baseline_class}" (the most common training class).</p>
      <table class="t"><tr><th>Test split</th><th>Images</th><th>mIoU (baseline)</th><th>Sand recall (baseline)</th><th>Sand IoU</th>
      <th>Big-rock recall / precision</th><th>Pixel acc (baseline)</th></tr>${rows}</table>`;
  } catch (e) { document.getElementById("ai-cards").innerHTML = `<div class="mw-err">Could not load AI results: ${e.message}</div>`; }
  if (meta.github) document.getElementById("gh").innerHTML = `· <a href="${meta.github}">Source on GitHub</a>`;
})();
