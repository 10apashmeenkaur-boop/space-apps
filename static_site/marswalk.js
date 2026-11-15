/* Marswalk route analysis in the browser - a line-by-line port of src/marswalk.py.
   Works in browsers (window.MW) and in Node (module.exports) for the parity test. */
(function (root) {
  const MARS_R = 3396190.0, SAMPLE_M = 20, STOP_RADIUS_M = 1500, AI_RADIUS_M = 200, AI_SHOW = 5;
  const PROVEN_M = 100, FLAT_KMH = 3.0, STOP_MIN = 15;
  let D = null;  // {hirise:{slope,elev,h,w,bounds}, mola:{...}, layers, ai}

  const rad = d => d * Math.PI / 180;
  function dist(lat1, lon1, lat2, lon2) {
    const p1 = rad(lat1), p2 = rad(lat2), dp = p2 - p1, dl = rad(lon2 - lon1);
    const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
    return 2 * MARS_R * Math.asin(Math.sqrt(Math.min(Math.max(a, 0), 1)));
  }
  function densify(pts) {
    const out = [pts[0]];
    for (let i = 0; i < pts.length - 1; i++) {
      const [a, b] = pts[i], [c, d] = pts[i + 1];
      const n = Math.max(1, Math.ceil(dist(a, b, c, d) / SAMPLE_M));
      for (let k = 1; k <= n; k++) out.push([a + (c - a) * k / n, b + (d - b) * k / n]);
    }
    return out;
  }
  function sample(g, lat, lon) {  // g: {arr,h,w,bounds:[s,n,w,e]}
    const [s, n, w, e] = g.bounds;
    const inside = lat >= s && lat <= n && lon >= w && lon <= e;
    let r = Math.trunc((n - lat) / (n - s) * g.h), c = Math.trunc((lon - w) / (e - w) * g.w);
    r = Math.min(Math.max(r, 0), g.h - 1); c = Math.min(Math.max(c, 0), g.w - 1);
    return {inside, i: r * g.w + c};
  }
  const tobler = g => FLAT_KMH * Math.exp(-3.5 * Math.abs(g + 0.05)) / Math.exp(-3.5 * 0.05);
  function slopeLevel(f15, f25) {
    if (f25 >= 0.02 || f15 >= 0.20) return "HIGH";
    if (f25 > 0 || f15 >= 0.05) return "MEDIUM";
    return "LOW";
  }
  function median(a) { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

  function analyse(points) {
    const pts = points.map(p => [+p[0], +p[1]]);
    if (pts.length < 2) return {error: "Click at least two waypoints on the map."};
    const P = densify(pts), N = P.length;
    const slp = new Float64Array(N), elev = new Float64Array(N), src = new Array(N);
    for (let i = 0; i < N; i++) {
      const [la, lo] = P[i];
      const h = sample(D.hirise, la, lo), m = sample(D.mola, la, lo);
      if (h.inside) { slp[i] = D.hirise.slope[h.i]; elev[i] = D.hirise.elev[h.i]; src[i] = "HiRISE"; }
      else if (m.inside) { slp[i] = D.mola.slope[m.i]; elev[i] = D.mola.elev[m.i]; src[i] = "MOLA"; }
      else return {error: "Part of the route is outside the Gale Crater data window (135.5–140°E, 3–7.5°S)."};
    }
    const seg = [], dz = [], mid = [], segSrc = [];
    let total = 0, hours = 0, climb = 0, desc = 0;
    for (let i = 0; i < N - 1; i++) {
      const s = dist(P[i][0], P[i][1], P[i + 1][0], P[i + 1][1]);
      const z = src[i] === src[i + 1] ? elev[i + 1] - elev[i] : 0;
      seg.push(s); dz.push(z); mid.push(Math.max(slp[i], slp[i + 1]));
      segSrc.push(src[i] === "HiRISE" && src[i + 1] === "HiRISE" ? "HiRISE" : "MOLA");
      total += s; hours += s / 1000 / tobler(z / Math.max(s, 1));
      if (z > 0) climb += z; else desc -= z;
    }
    const legs = []; let k = 0;
    for (let i = 0; i < pts.length - 1; i++) {
      const n = Math.max(1, Math.ceil(dist(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]) / SAMPLE_M));
      let L = 0, hi = 0, mx = -1;
      for (let j = k; j < k + n; j++) { L += seg[j]; if (segSrc[j] === "HiRISE") hi += seg[j]; mx = Math.max(mx, mid[j]); }
      const f = L ? hi / L : 0;
      legs.push({leg: i + 1, km: L / 1000, hirise_frac: f, max_slope: mx,
                 source: f > 0.999 ? "HiRISE 1 m" : (f < 0.001 ? "MOLA 463 m" : "mixed")});
      k += n;
    }
    const T = D.layers.traverse, stepT = Math.max(1, Math.floor(N / 400));
    let near = 0, cnt = 0;
    for (let i = 0; i < N; i += stepT) {
      const [a, b] = P[i]; let best = 1e9;
      for (const t of T) if (Math.abs(t[0] - a) < 0.01 && Math.abs(t[1] - b) < 0.01) best = Math.min(best, dist(a, b, t[0], t[1]));
      if (best <= PROVEN_M) near++; cnt++;
    }
    const minDist = (la, lo) => { let b = 1e12; for (const p of P) b = Math.min(b, dist(la, lo, p[0], p[1])); return b; };
    const stops = [];
    for (const st of D.layers.stops) { const d = minDist(st.lat, st.lon); if (d <= STOP_RADIUS_M) stops.push({...st, distance_m: d}); }
    stops.sort((a, b) => a.distance_m - b.distance_m);
    const science = stops.filter(s => s.kind !== "Rover position");
    const ai = [];
    const latMin = Math.min(...P.map(p => p[0])), latMax = Math.max(...P.map(p => p[0]));
    const lonMin = Math.min(...P.map(p => p[1])), lonMax = Math.max(...P.map(p => p[1]));
    for (const im of D.ai) {
      if (im.lat < latMin - 0.005 || im.lat > latMax + 0.005 || im.lon < lonMin - 0.005 || im.lon > lonMax + 0.005) continue;
      const d = minDist(im.lat, im.lon); if (d <= AI_RADIUS_M) ai.push({...im, distance_m: d});
    }
    ai.sort((a, b) => a.distance_m - b.distance_m);
    let f15 = 0, f25 = 0, hiKm = 0;
    for (let i = 0; i < seg.length; i++) { if (mid[i] > 15) f15 += seg[i]; if (mid[i] > 25) f25 += seg[i]; if (segSrc[i] === "HiRISE") hiKm += seg[i]; }
    f15 /= total; f25 /= total; hiKm /= 1000;
    const step = Math.max(1, Math.floor(N / 120)), km = [0];
    for (const s of seg) km.push(km[km.length - 1] + s);
    return {
      waypoints: pts, distance_km: total / 1000, hirise_km: hiKm, mola_km: total / 1000 - hiKm, legs,
      climb_m: climb, descent_m: desc, max_slope_deg: Math.max(...slp), median_slope_deg: median(slp),
      frac_gt15: f15, frac_gt25: f25, slope_risk: slopeLevel(f15, f25), proven_frac: near / cnt,
      ai, stops, walk_hours: hours, total_hours: hours + STOP_MIN * science.length / 60,
      profile: {km: km.filter((_, i) => i % step === 0), elev: Array.from(elev).filter((_, i) => i % step === 0)},
    };
  }

  const pct = v => Math.round(100 * v) + "%";
  const fmtH = h => `${Math.floor(h)} h ${String(Math.round((h % 1) * 60)).padStart(2, "0")} min`;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const RANK = {"GO": 0, "GO WITH CAUTION": 1, "HOLD": 2};

  function aiHTML(ai) {
    if (!ai.length) return `<li><b>AI sand/rock risk: unknown here</b> - no analysed AI4Mars photo lies within ${AI_RADIUS_M} m of this route.</li>`;
    const sand = ai.map(a => a.ai.sand), rock = ai.map(a => a.ai.bigrock);
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    const byStop = new Map();
    for (const a of ai) { const key = a.site + "/" + a.drive; if (!byStop.has(key)) byStop.set(key, []); byStop.get(key).push(a); }
    let rows = "";
    for (const [, ph] of [...byStop].slice(0, AI_SHOW)) {
      const gt = ph.filter(p => p.expert).map(p => p.expert);
      const gts = gt.length ? `; ground truth sand ${pct(mean(gt.map(g => g.sand)))}, big rock ${pct(mean(gt.map(g => g.bigrock)))}` : "";
      const worst = ph.map(p => p.verdict).reduce((a, b) => RANK[b] > RANK[a] ? b : a);
      rows += `<li>${Math.round(ph[0].distance_m)} m away - sol ${ph[0].sol_min}, site ${ph[0].site}/drive ${ph[0].drive} ` +
        `(${ph[0].loc}, ${ph.length} photo${ph.length > 1 ? "s" : ""}): AI sand <b>${pct(mean(ph.map(p => p.ai.sand)))}</b>, ` +
        `big rock <b>${pct(mean(ph.map(p => p.ai.bigrock)))}</b>${gts} - worst ${worst}</li>`;
    }
    const worst = ai.reduce((a, b) => (b.ai.sand + b.ai.bigrock) > (a.ai.sand + a.ai.bigrock) ? b : a);
    return `<li><b>AI sand/rock risk:</b> ${ai.length} analysed photos at ${byStop.size} rover stops within ${AI_RADIUS_M} m. ` +
      `Near-field sand mean <b>${pct(mean(sand))}</b> (max ${pct(Math.max(...sand))}), big rock mean <b>${pct(mean(rock))}</b> ` +
      `(max ${pct(Math.max(...rock))}). Riskiest photo: sol ${worst.sol_min}, ${pct(worst.ai.sand + worst.ai.bigrock)} sand + rock.` +
      `<br>Nearest rover stops with photos:<ul class='mw-dim'>${rows}</ul></li>`;
  }

  function reportHTML(r) {
    if (r.error) return `<div class='mw-err'>${esc(r.error)}</div>`;
    const col = {LOW: "#1f9d55", MEDIUM: "#c27c0e", HIGH: "#c53030"}[r.slope_risk];
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    const aiSand = r.ai.length ? `${pct(mean(r.ai.map(a => a.ai.sand)))} / ${pct(mean(r.ai.map(a => a.ai.bigrock)))}` : "n/a";
    const legs = r.legs.map(g => `<tr><td>${g.leg}</td><td>${g.km.toFixed(2)} km</td><td>${g.source}` +
      (g.source === "mixed" ? ` (${pct(g.hirise_frac)} HiRISE)` : "") + `</td><td>${Math.round(g.max_slope)}°</td></tr>`).join("");
    const stops = r.stops.map(s => `<li><b>${esc(s.name)}</b> - ${(s.distance_m / 1000).toFixed(2)} km from route ` +
      `<span class='mw-dim'>(${esc(s.kind)}; ${esc(s.source)})</span></li>`).join("") || "<li>None within 1.5 km.</li>";
    const km = r.profile.km.map(k => k / 1000), el = r.profile.elev, W = 320, H = 70;
    let svg = "";
    if (km.length > 1 && km[km.length - 1] > 0) {
      const lo = Math.min(...el), hi = Math.max(...el), span = Math.max(hi - lo, 1);
      const p = km.map((k, i) => `${(4 + (W - 8) * k / km[km.length - 1]).toFixed(1)},${(H - 6 - (H - 12) * (el[i] - lo) / span).toFixed(1)}`).join(" ");
      svg = `<svg viewBox='0 0 ${W} ${H}' width='100%' height='${H}' role='img' aria-label='elevation profile'>` +
        `<polyline points='${p}' fill='none' stroke='#2b6cb0' stroke-width='2'/>` +
        `<text x='${W - 4}' y='12' font-size='10' text-anchor='end' fill='currentColor'>${Math.round(hi)} m</text>` +
        `<text x='${W - 4}' y='${H - 2}' font-size='10' text-anchor='end' fill='currentColor'>${Math.round(lo)} m</text></svg>`;
    }
    const wp = r.waypoints.map(p => `(${p[0].toFixed(4)}, ${p[1].toFixed(4)})`).join(" → ");
    return `<div class='mw-report'><div class='mw-kpis'>
      <div><span>Distance</span><b>${r.distance_km.toFixed(2)} km</b></div>
      <div><span>Walk time (est.)</span><b>${fmtH(r.total_hours)}</b></div>
      <div><span>Slope risk</span><b style='color:${col}'>${r.slope_risk}</b></div>
      <div><span>Route on &gt;15° / &gt;25°</span><b>${pct(r.frac_gt15)} / ${pct(r.frac_gt25)}</b></div>
      <div><span>AI sand / big rock (200 m)</span><b>${aiSand}</b></div>
      <div><span>On Curiosity's path</span><b>${pct(r.proven_frac)}</b></div>
      <div><span>Climb / descent</span><b>+${Math.round(r.climb_m)} / -${Math.round(r.descent_m)} m</b></div>
      <div><span>Max slope</span><b>${Math.round(r.max_slope_deg)}°</b></div></div>${svg}
      <ul><li>Slope: ${pct(r.frac_gt15)} of the route crosses &gt;15° cells and ${pct(r.frac_gt25)} crosses &gt;25° cells
      (median ${Math.round(r.median_slope_deg)}°). Sources: <b>HiRISE 1 m DEM for ${r.hirise_km.toFixed(2)} km</b>
      (4 m slope baseline, steepest patch per 20 m cell), <b>MOLA 463 m for ${r.mola_km.toFixed(2)} km</b>
      (smooths slopes - hazards can be missed).
      <table class='mw-legs'><tr><th>Leg</th><th>Length</th><th>Slope source</th><th>Max</th></tr>${legs}</table></li>
      ${aiHTML(r.ai)}
      <li>${pct(r.proven_frac)} of this route lies within 100 m of ground Curiosity already drove (proven trafficable).</li>
      <li>Walk time = ${fmtH(r.walk_hours)} walking (3 km/h on flat, Tobler slope penalty) + ${STOP_MIN} min per science stop.</li></ul>
      <div><b>Science stops near the route</b><ul>${stops}</ul></div>
      <div class='mw-dim'>Waypoints (lat, lon): ${wp}</div></div>`;
  }

  const MW = {init: d => { D = d; }, analyse, reportHTML};
  if (typeof module !== "undefined" && module.exports) module.exports = MW; else root.MW = MW;
})(typeof window !== "undefined" ? window : globalThis);
