"""The static web version (web/marswalk.js) must give the same route numbers as src/marswalk.py."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.marswalk import analyse  # noqa: E402

WEB = ROOT / "web"
ROUTES = [
    [[-4.5895, 137.4416], [-4.6268, 137.4069], [-4.6380, 137.4015], [-4.6634, 137.3776]],  # HiRISE only
    [[-4.6380, 137.4015], [-4.95, 137.45]],                                                # HiRISE + MOLA
]

NODE = r"""
const fs = require('fs'), path = require('path'), MW = require(path.join(process.argv[1], 'marswalk.js'));
const d = p => path.join(process.argv[1], 'data', p), rd = (p, T) => { const b = fs.readFileSync(d(p)); return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };
const meta = JSON.parse(fs.readFileSync(d('grids.json')));
MW.init({hirise: {...meta.hirise, slope: rd('hirise_slope.u8', Uint8Array), elev: rd('hirise_elev.i16', Int16Array)},
         mola: {...meta.mola, slope: rd('mola_slope.f32', Float32Array), elev: rd('mola_elev.i16', Int16Array)},
         layers: JSON.parse(fs.readFileSync(d('layers.json'))), ai: JSON.parse(fs.readFileSync(d('ai_images.json')))});
const routes = JSON.parse(process.argv[2]);
console.log(JSON.stringify(routes.map(r => { const a = MW.analyse(r); a.ai = a.ai.length; a.stops = a.stops.length; delete a.profile; return a; })));
"""


@pytest.mark.skipif(not (WEB / "marswalk.js").exists() or shutil.which("node") is None, reason="needs web/ and node")
def test_js_matches_python():
    out = subprocess.run(["node", "-e", NODE, str(WEB), json.dumps(ROUTES)], capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    for route, j in zip(ROUTES, js):
        p = analyse(route)
        for k in ["distance_km", "hirise_km", "mola_km", "climb_m", "descent_m", "frac_gt15", "frac_gt25",
                  "max_slope_deg", "proven_frac", "walk_hours"]:
            assert j[k] == pytest.approx(p[k], rel=1e-3, abs=1e-3), k
        assert j["slope_risk"] == p["slope_risk"]
        assert j["ai"] == len(p["ai"]) and j["stops"] == len(p["stops"])
        assert [g["source"] for g in j["legs"]] == [g["source"] for g in p["legs"]]
