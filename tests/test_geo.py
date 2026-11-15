import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geolocate import locate, parse_name  # noqa: E402
from src.marswalk import analyse  # noqa: E402

WPS = [
    {"site": 60, "drive": 684, "lat": -4.70, "lon": 137.40, "sol": 1580},
    {"site": 60, "drive": 888, "lat": -4.71, "lon": 137.41, "sol": 1582},
    {"site": 60, "drive": 1088, "lat": -4.73, "lon": 137.43, "sol": 1583},
]


def test_parse_navcam_name():
    m = parse_name("NLB_537932733EDR_F0600888NCAM00320M1.JPG")
    assert m == {"camera": "NLB", "sclk": 537932733, "site": 60, "drive": 888}
    assert parse_name("not_a_navcam_name.png") is None


def test_locate_exact_and_interpolated():
    lat, lon, q, sols = locate(60, 888, WPS)
    assert (lat, lon, q, sols) == (-4.71, 137.41, "exact", (1582, 1582))
    lat, lon, q, sols = locate(60, 988, WPS)  # halfway between drives 888 and 1088
    assert q == "interpolated" and abs(lat + 4.72) < 1e-9 and abs(lon - 137.42) < 1e-9 and sols == (1582, 1583)
    assert locate(99, 1, WPS) is None


def test_route_analysis_uses_hirise_and_ai():
    r = analyse([[-4.5895, 137.4416], [-4.6268, 137.4069], [-4.6380, 137.4015], [-4.6634, 137.3776]])
    assert "error" not in r
    assert 5 < r["distance_km"] < 7 and r["slope_risk"] in {"LOW", "MEDIUM", "HIGH"}
    assert len(r["ai"]) > 0 and all(a["distance_m"] <= 200 for a in r["ai"])
    assert [a["distance_m"] for a in r["ai"]] == sorted(a["distance_m"] for a in r["ai"])
    assert r["hirise_km"] == r["distance_km"] and r["mola_km"] == 0
    assert all(g["source"] == "HiRISE 1 m" for g in r["legs"])
    assert any(s["name"] == "Kimberley" for s in r["stops"])


def test_route_leaving_hirise_falls_back_to_mola():
    r = analyse([[-4.6, 137.4], [-5.5, 137.4]])
    assert "error" not in r
    assert 0 < r["hirise_km"] < r["distance_km"] and r["mola_km"] > 0
    assert r["legs"][0]["source"] == "mixed"


def test_route_outside_gale_window_is_refused():
    assert "outside the Gale Crater data window" in analyse([[-4.6, 137.4], [-9.0, 137.4]])["error"]
