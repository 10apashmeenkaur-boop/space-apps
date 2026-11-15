"""7-slide deck showing the live Marswalk Planner: python docs/make_demo_deck.py"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SHOTS = DOCS / "shots"
LIVE = "https://marswalk-planner-jet.vercel.app"
GH = "https://github.com/10apashmeenkaur-boop/space-apps"
M = json.loads((ROOT / "results" / "metrics.json").read_text())

WHITE, MUTED, CYAN, MAGENTA = RGBColor(0xF2, 0xF4, 0xF8), RGBColor(0xB6, 0xBE, 0xCC), RGBColor(0x5E, 0xE7, 0xFF), RGBColor(0xFF, 0x5C, 0xD6)
PANEL = RGBColor(0x14, 0x18, 0x24)


def starfield(path, w=1920, h=1080, seed=7):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (6, 8, 16))
    glow = Image.new("RGB", (w, h), (0, 0, 0))
    g = ImageDraw.Draw(glow)
    g.ellipse((-300, -350, 700, 450), fill=(0, 70, 95))       # cyan glow, top-left
    g.ellipse((1350, 650, 2300, 1450), fill=(85, 0, 70))      # magenta glow, bottom-right
    glow = glow.filter(ImageFilter.GaussianBlur(220))
    img = Image.blend(img, glow, 0.55)
    d = ImageDraw.Draw(img)
    for _ in range(900):
        x, y, b = rng.integers(0, w), rng.integers(0, h), int(rng.integers(90, 255))
        r = 1 if rng.random() < 0.93 else 2
        d.ellipse((x, y, x + r, y + r), fill=(b, b, min(255, b + 15)))
    img.save(path, quality=92)


def tb(slide, x, y, w, h, text, size=16, color=WHITE, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Segoe UI"
        p.space_after = Pt(6)
    return box


def panel(slide, x, y, w, h):
    s = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = PANEL
    s.line.color.rgb = RGBColor(0x2A, 0x3A, 0x55)
    s.shadow.inherit = False
    return s


def shot(slide, name, x, y, w, caption):
    pic = slide.shapes.add_picture(str(SHOTS / name), Inches(x), Inches(y), width=Inches(w))
    pic.line.color.rgb = CYAN
    pic.line.width = Pt(1.25)
    h = pic.height / 914400
    tb(slide, x, y + h + 0.05, w, 0.35, caption, size=12, color=MUTED)
    return pic


def base(prs, bg, title, kicker):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.shapes.add_picture(str(bg), 0, 0, width=prs.slide_width, height=prs.slide_height)
    tb(s, 0.5, 0.22, 12.3, 0.4, kicker, size=13, color=CYAN, bold=True)
    tb(s, 0.5, 0.5, 12.3, 0.8, title, size=30, bold=True)
    return s


def bullets(s, x, y, w, h, items, size=15):
    panel(s, x, y, w, h)
    tb(s, x + 0.2, y + 0.15, w - 0.4, h - 0.3, ["•  " + i for i in items], size=size)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def table(s, x, y, w, rows, col_w, size=13, header_color=CYAN):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.38 * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor(0x1C, 0x22, 0x33) if i else RGBColor(0x10, 0x14, 0x20)
            c.text = str(val)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.name = Pt(size), "Segoe UI"
                    r.font.color.rgb = header_color if i == 0 else WHITE
                    r.font.bold = i == 0
    return t


def main():
    bg = DOCS / "_starfield.jpg"
    starfield(bg)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1 - title
    s = base(prs, bg, "Marswalk Planner: plan a safe, science-rich walk on Mars", "NASA SPACE APPS 2026 · INTERPLANETARY SURVIVAL GUIDE: MARTIAN MAP")
    shot(s, "1_map.png", 0.5, 1.45, 8.0, "Live site: Gale Crater, Curiosity's field site, with NASA layers and Curiosity's real path")
    bullets(s, 8.8, 1.45, 4.05, 4.4, [
        "One map of Gale Crater built from several NASA missions",
        "Click a route → slope risk, sand/rock risk, science stops, walking time",
        "319 real Curiosity photos on the map, checked by our AI",
        "Runs in any browser; no install"], size=15)
    tb(s, 8.8, 6.0, 4.1, 1.0, ["Live: " + LIVE, "Code: " + GH], size=13, color=CYAN)
    notes(s, "Hi! This is Marswalk Planner. Imagine you're one of the first people to walk on Mars. Nobody on Earth can "
             "help you in real time, because a radio signal takes between 4 and 22 minutes one way (that's from JPL's own "
             "rover drivers). So you need to plan your walk well before you step out. Our page puts the NASA data you need "
             "on one map of Gale Crater, where Curiosity has been driving since 2012. You click a route, and it tells you how "
             "steep it is, whether there's sand, what science is nearby and how long it will take. It's live at the link on "
             "the slide, so you can try it yourself.")

    # 2 - layers
    s = base(prs, bg, "One map, many NASA missions", "STEP 1 · THE LAYERS")
    shot(s, "2_layers.png", 0.5, 1.45, 8.0, "Orange/red = slopes >15°/>25° from the HiRISE 1 m DEM · white line = Curiosity's traverse · dots = AI-checked photos")
    table(s, 8.8, 1.45, 4.05, [
        ["Layer", "NASA source"],
        ["Satellite images", "MRO CTX (Mars Trek)"],
        ["Slope hazards", "MRO HiRISE 1 m DEM"],
        ["Elevation", "MGS MOLA"],
        ["Clay / sulfate", "MRO CRISM"],
        ["Water minerals", "CRISM + OMEGA"],
        ["Rover path, stops", "Curiosity (NASA MMGIS)"],
        ["Rover photos", "Curiosity Navcam (AI4Mars)"]], [1.75, 2.3], size=12)
    tb(s, 8.8, 4.75, 4.05, 1.6, ["Toggle any layer from the top-right of the map.",
                                 "Slope uses HiRISE where it exists and MOLA outside, and the report says which."],
       size=13, color=MUTED)
    notes(s, "First, the layers. The grey background is a satellite mosaic from the Mars Reconnaissance Orbiter's CTX "
             "camera. The orange and red patches are slopes we computed from a 1-metre HiRISE elevation model: orange is "
             "steeper than 15 degrees, red steeper than 25. Then there's MOLA elevation, mineral signals from CRISM, and "
             "Curiosity's real path from NASA's own mission map. Every layer comes from a NASA mission, and the table shows which.")

    # 3 - route
    s = base(prs, bg, "Click a route, get an instant briefing", "STEP 2 · PLAN A WALK")
    shot(s, "3_route.png", 0.5, 1.45, 8.0, "Route A: Bradbury Landing → Kimberley → Pahrump Hills → Mary Anning drill sites (9.78 km)")
    table(s, 8.8, 1.45, 4.05, [
        ["Route A result", "Value"],
        ["Distance", "9.78 km"],
        ["Walk time (est.)", "5 h 18 min"],
        ["Slope risk", "HIGH (18% >15°)"],
        ["AI sand / big rock", "11% / 0%"],
        ["On Curiosity's path", "41%"],
        ["Climb / descent", "+529 / −144 m"]], [2.0, 2.05], size=12)
    tb(s, 8.8, 4.35, 4.05, 2.0, ["Every leg says which elevation data it used (HiRISE or MOLA).",
                                 "Walk time = 3 km/h on flat ground, slowed on slopes, + 15 min per science stop: an estimate."],
       size=13, color=MUTED)
    notes(s, "Here's the main feature. Press Draw route and click points on the map. I picked a science walk from where "
             "Curiosity landed to the Kimberley and Pahrump Hills sites and on to the Mary Anning drill sites. In a split "
             "second it says: about 9.8 kilometres, about 5 hours, and HIGH slope risk, because 18% of the route crosses "
             "steep patches. It also counts how much of the walk is on ground Curiosity already drove, which we know is "
             "passable. The walk time is an estimate, not a validated spacesuit model.")

    # 4 - popup
    s = base(prs, bg, "Real rover photos, checked by AI, on the map", "STEP 3 · SEE THE GROUND")
    shot(s, "4_popup.png", 0.5, 1.45, 8.0, "Click a dot: Curiosity Navcam photo | AI prediction | expert labels (NASA/JPL-Caltech AI4Mars)")
    bullets(s, 8.8, 1.45, 4.05, 4.9, [
        "Each Navcam filename hides the rover's location: F0600888 = site 60, drive 888",
        "We matched 319 photos to NASA's waypoints (287 exact)",
        "Dot colour = predicted sand + big rock: green <10%, red >50%",
        "Popup shows our AI next to the experts' labels"], size=14)
    notes(s, "This is my favourite part. NASA's AI4Mars test photos don't come with map coordinates, but every filename "
             "hides a code for where the rover was: site and drive numbers. We decoded them and matched them to NASA's "
             "published waypoints, which put 319 real photos on the map. Click a dot and you see the photo, what our AI "
             "thinks is sand or rock, and what the human experts labelled, side by side. In this one the AI says 73% sand "
             "and the experts said 86%: close, but not perfect.")

    # 5 - AI tab
    s = base(prs, bg, "AI ground check: is this patch sand?", "STEP 4 · THE AI")
    shot(s, "5_ai.png", 0.5, 1.45, 8.0, "AI tab: 8 test photos → terrain map + safest path + GO / CAUTION / HOLD (precomputed on the web version)")
    bullets(s, 8.8, 1.45, 4.05, 4.9, [
        "Colours: soil brown, bedrock grey, sand yellow, big rock red",
        "Green line = safest path through the nearby ground (A*)",
        "Model: DINOv2-small features + LightGBM, CPU only, ~0.5 s per photo",
        "Web version shows saved results; the full app runs it live"], size=14)
    notes(s, "Orbital maps can't tell you if one metre of ground is soft sand, so we trained an AI on NASA and JPL's "
             "labelled rover photos. It colours each part of the photo as soil, bedrock, sand or big rock, then draws the "
             "safest path through the near ground and gives a verdict: GO, CAUTION or HOLD. It runs on a laptop CPU. On "
             "this free web version the results are precomputed, and the app in our GitHub repo runs it live on your own photos.")

    # 6 - accuracy
    s = base(prs, bg, "How accurate is it? Tested on NASA's own expert set", "STEP 5 · RESULTS")
    rows = [["Test split", "mIoU (baseline)", "Sand recall (baseline)", "Pixel acc (baseline)"]]
    for k in ["test_min3", "test_min2", "test_min1"]:
        r = M["splits"][k]
        m, b = r["model"], r["baseline"]
        rows.append([k, f"{m['mIoU']:.3f} ({b['mIoU']:.3f})", f"{100 * m['per_class']['sand']['recall']:.1f}% "
                     f"({100 * b['per_class']['sand']['recall']:.0f}%)", f"{100 * m['pixel_accuracy']:.1f}% ({100 * b['pixel_accuracy']:.1f}%)"])
    table(s, 0.5, 1.5, 7.6, rows, [1.6, 2.0, 2.1, 1.9], size=13)
    s.shapes.add_picture(str(ROOT / "results" / "confusion_matrix.png"), Inches(8.6), Inches(1.45), height=Inches(3.6))
    tb(s, 8.6, 5.1, 4.3, 0.4, "Confusion matrix, test_min3 (rows = expert label)", size=12, color=MUTED)
    bullets(s, 0.5, 3.35, 7.6, 3.5, [
        "322 expert-labelled Curiosity photos the model never saw in training",
        "Baseline = always guess the commonest class (bedrock): it never finds sand",
        "test_min3 keeps only pixels all labellers agreed on, so it is the easiest split",
        "Weak spot: big rocks are rare and over-predicted (precision 0.11–0.26)",
        "JPL's published models (DeepLabv3 >96%, SPOCv2 91.41%) used a different setup; we don't claim to beat them"], size=14)
    notes(s, "Does it work? We tested on NASA's official expert test set: 322 photos the model never saw. Two quick "
             "explanations. mIoU, mean intersection over union, measures how well the coloured areas overlap the experts' "
             "areas, averaged over the four terrain types; 1 would be perfect. We get 0.78, against 0.06 for a dumb guess. "
             "Sand recall answers: of all the real sand, how much did we catch? 96%. The honest weak spot is big rocks: "
             "they're rare, and the model flags too many. JPL's own models used a different setup, so we don't claim to beat them.")

    # 7 - data, tests, links
    s = base(prs, bg, "Checked, tested, and live", "STEP 6 · TRUST + LINKS")
    shot(s, "7_data.png", 0.5, 1.45, 6.4, "Data & limits tab: every source and every limit is on the page")
    table(s, 7.2, 1.45, 5.65, [
        ["Check", "Result"],
        ["Automated tests", "17 / 17 pass"],
        ["Web vs Python route engine", "same numbers (tested)"],
        ["Train/test leakage", "none (pixel hashes)"],
        ["Filename → location", "55_940 → sol 1384, exact"],
        ["Fresh install, no data folder", "runs"]], [3.0, 2.65], size=12)
    bullets(s, 7.2, 4.0, 5.65, 2.3, [
        "Next: helmet-camera ground check, route modes (safest / flattest / emergency return)",
        "Next: suit time, return-to-base and radiation dose per walk (0.67 mSv/day, Curiosity RAD)"], size=13)
    tb(s, 0.5, 6.35, 12.3, 0.9, ["Try it: " + LIVE, "Code + README: " + GH,
                                 "Credits: NASA/JPL-Caltech · NASA/JPL/MSSS/The Murray Lab · NASA/JPL/UArizona · NASA Mars Trek · NASA MMGIS"],
       size=12, color=CYAN)
    notes(s, "Finally, trust. Every data source and limit is written on the page itself. We have 17 automated tests, "
             "including one that proves the web version gives the same numbers as our Python code. We checked there's no "
             "overlap between training and test photos, and that our location decoding matches NASA's waypoints exactly. "
             "Next, we'd like a helmet-camera version of the ground check, route modes such as an emergency return when "
             "oxygen is low, and a radiation estimate per walk using Curiosity's measured 0.67 millisieverts per day. "
             "The links are on the slide. Thanks!")

    out = DOCS / "Marswalk_Planner_Demo.pptx"
    prs.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
