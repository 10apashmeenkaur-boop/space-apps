"""Sand Trap Detector - Streamlit app.

Upload a Mars rover Navcam image (or pick a sample) -> terrain map -> safest
image-space route through the near field -> drive briefing.
"""
import io
import json
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

import config as C
from src.briefing import make_briefing
from src.infer import load_classifier, proba_grid, upsample
from src.planner import near_field_top, plan

st.set_page_config(page_title="Sand Trap Detector", page_icon="🛞", layout="wide")

VIEW = 640  # display resolution
PAL = np.array([C.COLORS[c] for c in C.CLASSES], dtype=np.float32)
LABELS = {"soil": "Soil", "bedrock": "Bedrock", "sand": "Sand (hazard)", "bigrock": "Big rock (hazard)"}


@st.cache_resource(show_spinner="Loading DINOv2 + terrain classifier (first run only)...")
def load_models():
    from src.features import DinoFeatures

    return DinoFeatures(), load_classifier()


@st.cache_data(show_spinner=False)
def load_metrics():
    p = C.RESULTS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False, max_entries=32)
def predict(img_bytes: bytes):
    grey = np.load(io.BytesIO(img_bytes))
    dino, clf = load_models()
    t = time.time()
    feats = dino([grey])[0]
    prob = proba_grid(clf, feats)
    return prob, time.time() - t


def read_image(data: bytes):
    """Return (grey uint8 HxW, list of warnings) or raise ValueError."""
    warnings = []
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("That file is not a readable image.") from e
    if img.mode not in ("L", "I;16", "I", "P", "1"):
        arr = np.asarray(img.convert("RGB"), dtype=np.int16)
        if np.abs(arr[..., 0] - arr[..., 1]).mean() > 3 or np.abs(arr[..., 1] - arr[..., 2]).mean() > 3:
            warnings.append("Colour image converted to grey.")
    grey = np.asarray(img.convert("L"), dtype=np.uint8)
    h, w = grey.shape
    if min(h, w) < 64:
        raise ValueError("Image is too small (need at least 64x64 pixels).")
    if abs(h - w) / max(h, w) > 0.05:
        warnings.append(f"Image is {w}x{h}; it was stretched to a square like Navcam frames.")
    if warnings or (h, w) != (C.ORIG, C.ORIG):
        warnings.append("The model was trained on Curiosity Navcam images; other cameras may be less accurate.")
    return grey, warnings


def to_npy_bytes(grey):
    b = io.BytesIO()
    np.save(b, grey)
    return b.getvalue()


def render_overlay(grey, prob, result, alpha=0.5):
    g = cv2.resize(grey, (VIEW, VIEW), interpolation=cv2.INTER_AREA).astype(np.float32)
    lab = upsample(prob, VIEW).argmax(-1)
    rgb = np.repeat(g[..., None], 3, -1)
    out = ((1 - alpha) * rgb + alpha * PAL[lab]).astype(np.uint8)
    n = result["prob"].shape[0]
    s = VIEW / n
    top = near_field_top(n)
    y = int(top * s)
    for x0 in range(0, VIEW, 24):
        cv2.line(out, (x0, y), (min(x0 + 12, VIEW - 1), y), (255, 255, 255), 2)
    cv2.putText(out, "near-field limit", (8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                cv2.LINE_AA)
    for r, c in np.argwhere(result["blocked"][top:]):
        r += top
        cv2.rectangle(out, (int(c * s), int(r * s)), (int((c + 1) * s), int((r + 1) * s)), (120, 0, 0), 1)

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
    chip = ("<span style='display:inline-flex;align-items:center;margin-right:16px;'>"
            "<span style='width:{w}px;height:{h}px;border-radius:3px;background:rgb{rgb};"
            "display:inline-block;margin-right:6px;'></span>{name}</span>")
    html = "".join(chip.format(w=14, h=14, rgb=C.COLORS[c], name=LABELS[c]) for c in C.CLASSES)
    html += chip.format(w=22, h=5, rgb=(40, 230, 90), name="Safe route")
    return f"<div style='font-size:0.9rem;line-height:2'>{html}</div>"


# ---------------- UI ----------------
st.title("🛞 Sand Trap Detector")
st.caption("Spirit's mission ended in a Martian sand trap in 2009. This tool reads a rover camera image, "
           "maps soil, bedrock, sand and big rocks, and previews the safest route through the near field.")

samples = sorted(C.SAMPLES_DIR.glob("*.png"))
left, right = st.columns([1, 2])
with left:
    source = st.radio("Image source", ["Sample test image", "Upload my own"], horizontal=True)
    raw = None
    if source == "Sample test image":
        if samples:
            name = st.selectbox("Sample (AI4Mars expert test set - never used in training)",
                                [p.name for p in samples])
            raw = (C.SAMPLES_DIR / name).read_bytes()
        else:
            st.info("No sample images found in samples/.")
    else:
        up = st.file_uploader("Navcam image (PNG/JPG, greyscale, ideally 1024x1024)",
                              type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        if up is not None:
            raw = up.getvalue()

if raw is None:
    st.stop()

try:
    grey, warns = read_image(raw)
except ValueError as e:
    st.warning(f"{e} Please try a different file.")
    st.stop()
for w in warns:
    st.warning(w)

with st.spinner("Classifying terrain..."):
    prob, secs = predict(to_npy_bytes(grey))

n = C.PLAN_GRID
top = near_field_top(n)
with left:
    st.markdown("**Route target**")
    auto = st.checkbox("Auto: farthest safe point straight ahead", value=True)
    tx = st.slider("Target across (left → right)", 0, 100, 50, disabled=auto)
    ty = st.slider("Target distance (near → far edge of near field)", 0, 100, 100, disabled=auto)
goal = None if auto else (int(round((n - 1) - ty / 100 * (n - 1 - top))), int(round(tx / 100 * (n - 1))))
result = plan(prob, goal)
brief = make_briefing(result)

with right:
    c1, c2 = st.columns(2)
    c1.image(cv2.resize(grey, (VIEW, VIEW), interpolation=cv2.INTER_AREA), caption="Original Navcam view",
             width="stretch")
    c2.image(render_overlay(grey, prob, result), caption="Terrain map + safe route (image-space preview)",
             width="stretch")
    st.markdown(legend_html(), unsafe_allow_html=True)

st.subheader("Drive briefing")
color = {"GO": "#1f9d55", "GO WITH CAUTION": "#d69e2e", "HOLD": "#c53030"}[brief["verdict"]]
b1, b2, b3, b4 = st.columns(4)
b1.markdown(f"<div style='font-size:1.6rem;font-weight:700;color:{color}'>{brief['verdict']}</div>",
            unsafe_allow_html=True)
b2.metric("Route risk score", "—" if brief["risk"] is None else f"{brief['risk']:.2f}",
          help="Mean P(sand) + P(big rock) along the route: 0 = clean ground, 1 = all hazard.")
b3.metric("Sand in near field", f"{brief['near_field_fractions']['sand']:.0%}")
b4.metric("Model time", f"{secs:.1f} s", help="DINOv2 features + classifier on CPU (cached on repeat).")
for line in brief["lines"]:
    st.markdown(f"- {line}")

with st.expander("How accurate is this?"):
    m = load_metrics()
    if m is None:
        st.info("metrics.json not found - run src/evaluate.py.")
    else:
        rows = []
        for s, r in m["splits"].items():
            mm, bb = r["model"], r["baseline"]
            rows.append({
                "Test split": s, "Images": r["n_images"],
                "Pixel accuracy": f"{mm['pixel_accuracy']:.1%}", "mIoU": f"{mm['mIoU']:.3f}",
                "Sand recall": f"{mm['per_class']['sand']['recall']:.1%}",
                "Sand IoU": f"{mm['per_class']['sand']['IoU']:.3f}",
                "Big-rock recall": f"{mm['per_class']['bigrock']['recall']:.1%}",
                f"Baseline acc (always {m['baseline_class']})": f"{bb['pixel_accuracy']:.1%}",
            })
        st.markdown(f"Scored pixel-by-pixel on the **official AI4Mars expert test set** "
                    f"(322 Curiosity images, never seen in training). Classifier: **{m['model']}** "
                    f"on frozen DINOv2-small patch features.")
        st.dataframe(rows, hide_index=True, width="stretch")
        cm_png = C.RESULTS_DIR / "confusion_matrix.png"
        cc1, cc2 = st.columns(2)
        if cm_png.exists():
            cc1.image(str(cm_png), caption="Confusion matrix, test_min3 (strictest agreement)")
        cc2.markdown(
            "**Reference points (published; protocol not identical):** JPL's DeepLabv3 on AI4Mars "
            "≈ 96% pixel accuracy; SPOCv2 91.4%. Those are fine-tuned end-to-end on GPUs; "
            "ours is a frozen foundation model + small classifier that runs on a laptop CPU.\n\n"
            "**Honest limits**\n"
            "- Big rocks are rare (~3% of labels) and the weakest class.\n"
            "- Labels only reach ~30 m, so the far field is not planned.\n"
            "- 32×32-pixel patches blur thin boundaries.\n"
            "- The route is drawn on the camera view, not a top-down map: a preview for operators, "
            "not distances in metres.\n"
            "- Trained on Curiosity Navcam only; other cameras may be less accurate.")

st.divider()
st.caption("Data: NASA/JPL-Caltech AI4Mars (Curiosity Navcam, CC0). "
           "Model: DINOv2-small (Meta) + LightGBM / Logistic Regression. Route: A* on a probability cost map.")
