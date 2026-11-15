"""Take 1600x900 screenshots of the live web version for the demo deck: python docs/shots.py"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://marswalk-planner-jet.vercel.app/"
OUT = Path(__file__).parent / "shots"
OUT.mkdir(exist_ok=True)
ROUTE_A = [[-4.589467, 137.441633], [-4.637998, 137.401476], [-4.663356, 137.377555], [-4.729296, 137.391119]]

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_function("window._mw !== undefined", timeout=60000)
    pg.wait_for_timeout(2500)
    pg.screenshot(path=str(OUT / "1_map.png"))

    # zoomed-out view: HiRISE slope + markers + traverse
    pg.evaluate("window._map.setView([-4.70, 137.42], 9)")
    pg.wait_for_timeout(3500)
    pg.screenshot(path=str(OUT / "2_layers.png"))

    # route A + report
    pg.evaluate(f"window._mw.setRoute({ROUTE_A})")
    pg.wait_for_timeout(3500)
    pg.screenshot(path=str(OUT / "3_route.png"))

    # 3-image popup on a high-risk marker
    pg.evaluate("window._map.setView([-4.665, 137.39], 12)")
    pg.wait_for_timeout(2500)
    pg.evaluate("""() => { const m = [...document.querySelectorAll('#map path.leaflet-interactive')]
        .filter(p => p.getAttribute('fill') === '#d73027'); m[0].dispatchEvent(new MouseEvent('click', {bubbles: true})); }""")
    pg.wait_for_timeout(3500)
    pg.screenshot(path=str(OUT / "4_popup.png"))

    # AI tab
    pg.click("button[data-t='t-ai']")
    pg.wait_for_timeout(2500)
    pg.screenshot(path=str(OUT / "5_ai.png"))
    pg.evaluate("document.getElementById('metrics').scrollIntoView()")
    pg.wait_for_timeout(1000)
    pg.screenshot(path=str(OUT / "6_metrics.png"))

    pg.click("button[data-t='t-data']")
    pg.wait_for_timeout(800)
    pg.screenshot(path=str(OUT / "7_data.png"))
    b.close()
print("saved", sorted(x.name for x in OUT.glob("*.png")))
