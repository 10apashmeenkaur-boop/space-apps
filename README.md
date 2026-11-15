# Marswalk Planner: Gale Crater, with an AI ground-check layer

**NASA Space Apps Challenge 2026 · Challenge: Interplanetary Survival Guide: Martian Map**
**Team:** Everything You Despise · **Built by:** Pashmeen Kaur

| | |
|---|---|
| **Code** | https://github.com/10apashmeenkaur-boop/space-apps |
| **Demo (4-slide deck)** | [docs/Marswalk_Planner.pdf](docs/Marswalk_Planner.pdf) |
| **Live demo (web version)** | https://marswalk-planner-jet.vercel.app : map, layers, route planner and AI photo markers run in the browser; AI tab shows precomputed results |
| **Run it yourself** | ~5 min on any laptop, no GPU: see [How to run](#how-to-run) |

The challenge asks for *a layered, integrated view of a location or route on the Martian surface
that pulls together data from multiple NASA science missions and could help a human explorer plan
and carry out a successful Marswalk while doing new and exciting science along the way.*

**AI ground check, official AI4Mars test set (`test_min3`): mIoU 0.777 vs 0.056 for a majority-class
baseline; sand recall 96.3% vs 0%.** Runs on a laptop CPU.

## In one minute

- **Why:** Mars is mostly unexplored up close, and the first walkers will be on their own: a
  radio signal takes 4–22 minutes each way. Steep slopes, soft sand (it ended NASA's Spirit rover
  in 2009) and big rocks are real dangers for someone in a spacesuit carrying tools.
- **What:** one map of Gale Crater, Curiosity's field site, built from several NASA missions. It
  combines slope hazards, mineral signals, Curiosity's real path and **319 real rover photos
  checked by our AI**. You click a route and get distance, climb, slope risk, sand/rock risk,
  science stops and walking time.
- **Proof:** the AI was trained on NASA/JPL's labelled rover photos and tested on NASA's own expert
  test set, which it never saw. It catches 96% of the sand. 16/16 automated tests pass.

## Demo: what to click

1. **Marswalk Map** tab → **★ Example route**. The route report fills in, for example
   5.82 km, ~3 h, MEDIUM slope risk, AI sand 11% from 102 nearby Curiosity photos.
2. Click a **red AI marker**. It shows the real Navcam photo, the AI terrain map and the experts' labels.
3. **Draw route** → click points (follow the white Curiosity line for AI sand data) → the report updates on every click.
4. **AI Ground Check** tab → click a sample (or drop a file from `samples/`): yellow = sand,
   red = big rock, green line = safest path, plus a GO / CAUTION / HOLD verdict.
5. **Data & limits** tab: every source and every limit.

### Map legend
| On the map | Meaning |
|---|---|
| Grey background | MRO CTX satellite mosaic of Gale Crater |
| Orange / red patches | Slope >15° / >25° (HiRISE 1 m DEM; MOLA outside its box) |
| White line | **Curiosity traverse**: the path the rover actually drove since 2012 (NASA MMGIS) |
| Green dashed line, numbered | **Your Marswalk**: the waypoints you clicked |
| Green / yellow / orange / red dots | AI-checked Curiosity photos: predicted sand + big rock <10 / 10–30 / 30–50 / >50% |
| Yellow dots | Curiosity science sites (landing site, Kimberley, Pahrump Hills, drill sites…) |
| Purple dots | Orbital water-mineral detections (CRISM + OMEGA) |
| Pink / blue | CRISM sulfate / clay signal strength (one observation) |

## The problem

A walk on Mars is planned on Earth, from orbit, long before anyone steps out. Two things end
excursions: **ground you cannot cross** (steep slopes, soft sand: sand ended the Spirit rover's
mission in 2009) and **ground not worth crossing** (a walk that misses the science). The data to
judge both exists, but it is scattered across missions, instruments and archives.

## What we built

**1. Marswalk Map (Gale Crater, Curiosity's field site)**: one Leaflet map with layers from
several NASA missions and archives:

| Layer | Mission / instrument | Source |
|---|---|---|
| CTX global mosaic (~6 m/px) | MRO Context Camera | NASA Mars Trek WMTS (Caltech Murray Lab beta01 mosaic) |
| MOLA colour hillshade (463 m/px) | MGS Mars Orbiter Laser Altimeter | NASA Mars Trek WMTS |
| Elevation and **slope hazards** (>15°, >25°) | MRO HiRISE stereo DEM of Gale, 1 m | NASA Mars Trek (`Gale_DEM_SMG_1m`) |
| **Clay / sulfate** band-parameter strength | MRO CRISM MTRDR FRT0000A091 (D2300, SINDEX2, BD1900_2) | PDS Geosciences Node |
| Hydrous mineral detections | MRO CRISM + Mars Express OMEGA | NASA Mars Trek |
| Curiosity traverse, 1,389 waypoints, science sites | MSL Curiosity | NASA MMGIS (mars.nasa.gov/maps) |
| **AI ground checks**: 319 Navcam images placed on the traverse | MSL Navcam + AI4Mars | site/drive in each filename, matched to MMGIS waypoints |

**HiRISE slope:** the 1 m DEM is sampled at 4 m (30 tiles from Trek's image server), slope is
computed at that 4 m baseline, and the **steepest patch in each 20 m cell** is kept, so small
scarps and crater walls stay visible. With MOLA's 463 m cells, only 4.4% of Gale was steeper
than 15°. At HiRISE scale, 22% of 20 m cells in the DEM area contain a >15° patch and 6.6% a
>25° patch.

**Placing rover images on the map:** every Curiosity Navcam product ID carries the rover's
location counter. For `NLB_537932733EDR_F0600888NCAM00320M1`, `F0600888` means site 060,
drive 0888. The AI4Mars parquet files keep these filenames inside the image records, so all 322
test images were recovered and matched to NASA MMGIS waypoints: 287 by exact site/drive, 30
interpolated by drive number within the site, and 2 at the nearest waypoint of the site. 3 could
not be placed because their site is not in the waypoint list. Cross-check: the sol implied by
each image's spacecraft clock falls inside the sol range of its matched site. The result is
**319 images at 89 rover stops** (sols 127–2434). Each one carries the AI's near-field
soil/bedrock/sand/big-rock fractions and the experts' labels, which differ by 9.7 points of
sand on average.
**Route tool:** click waypoints and get:
- distance, elevation profile, climb and descent;
- **slope risk**: the share of the route crossing >15° and >25° cells. HiRISE is used inside its box
  and MOLA (463 m) outside it, and each leg is labelled with the source it used (HiRISE / MOLA / mixed);
- **AI sand/rock risk**: the nearest analysed Navcam photos within 200 m of the route, each with its
  predicted and ground-truth sand and big-rock shares;
- how much of the walk follows ground Curiosity already drove (proven trafficable);
- **science stops** within 1.5 km;
- an **estimated walking time** (3 km/h on flat ground, Tobler's slope penalty, plus 15 min per
  science stop).

**2. AI Ground Check**: for the last-metre question orbital data cannot answer ("is *this* patch
sand?"). Upload a rover Navcam image and the model colours every part as **soil / bedrock / sand /
big rock**. Sand and big rock become hazards on a probability cost map, A* draws the safest route
through the near field, and a briefing gives a GO / CAUTION / HOLD verdict.

## AI method (CPU only)

1. Data: NASA/JPL AI4Mars (Curiosity Navcam, CC0), streamed from Hugging Face
   (`hassanjbara/AI4MARS`). 1,500 labelled train images (spread over all 13 shards; images with <5%
   labelled pixels skipped) and all 322 images of the official expert test set.
   Rover body and ground beyond 30 m are masked out.
2. Features: frozen **DINOv2-small** (Meta). Our own preprocessing (no `AutoImageProcessor`):
   resize to 448×448, grey → 3 channels, ImageNet normalisation → a 32×32 grid of 384-d patch features.
3. Cell labels: majority class of each 32×32-pixel cell with ≥60% labelled pixels, at most 250
   cells per image (class-balanced).
4. Classifiers: Logistic Regression and LightGBM, both class-balanced. Validation split **by image**
   (85/15), and the model is picked by validation mIoU: **LightGBM** (val mIoU 0.746 vs 0.697).
5. Test: class probabilities are bilinearly upsampled to 1024×1024 and scored pixel by pixel against
   the expert masks, ignoring unlabelled pixels.

**Leakage check:** every image is identified by an MD5 hash of its pixels, which also catches
the same photo stored twice under different names. No test image appears in train or validation
(asserted in code; ids in `results/image_ids.json`).

## Results (official AI4Mars test set, 322 images)

| Test split | **mIoU** (baseline) | **Sand recall** (baseline) | Sand IoU | Big-rock recall / precision | Pixel acc (baseline) |
|---|---|---|---|---|---|
| **test_min3** | **0.777** (0.056) | **96.3%** (0%) | 0.956 | 94% / 0.26 | 97.8% (22.4%) |
| test_min2 | 0.711 (0.068) | 93.7% (0%) | 0.920 | 81% / 0.11 | 95.5% (27.4%) |
| test_min1 | 0.665 (0.077) | 87.6% (0%) | 0.848 | 50% / 0.13 | 91.8% (30.9%) |

*Baseline* = always predict the most common **training** class (bedrock), so it never finds sand.
No test information is used. **`test_min3` scores only pixels that all of the (at least three)
labellers agreed on**, so it leaves out the ambiguous boundaries and is the easiest of the three
splits. `test_min2` and `test_min1` relax the agreement rule, and scores drop accordingly. Even a
constant guess of the most common *test* class (soil) would score at most 51% pixel accuracy.

![Confusion matrix](results/confusion_matrix.png)

**Published reference points (a different setup, so not a like-for-like comparison):** the
AI4Mars paper (CVPR 2021 workshop) reports **DeepLabv3 above 96% overall accuracy**, and the
Zooniverse AI4Mars results report **SPOCv2 at 91.41%**. Those models were trained and scored under
their own protocols. Ours scores pixels of the public test masks, and `test_min3` only scores
pixels where the labellers all agreed. **We do not claim to beat JPL's models.** The point here is
different: a frozen foundation model plus a small classifier, trained in minutes, gets usable
sand detection on a laptop CPU (~0.5 s per image).

Good and failure examples are in `results/examples/`. The worst cases are deep shadows on dune
fields, which get read as bedrock or big rock.

## Limits (read before trusting a route)

- **Big rocks are rare** (0.9% of training pixels, 0.1–0.2% of test pixels). Recall is high but
  precision is low (0.11–0.26): the model over-calls big rocks, so the planner is conservative, not
  blind.
- **Slope covers the HiRISE DEM area only** (137.12–137.73°E, 4.25–4.93°S, around Curiosity's
  traverse). Outside it the route falls back to MOLA 463 m slope, which smooths hazards away;
  the report labels every leg with its source, so treat MOLA legs as unverified. Each 20 m HiRISE
  cell keeps its steepest 4 m patch, so a flagged cell may be mostly walkable. Mosaic seams in the
  DEM can create false steep lines.
- **CRISM layers show band-parameter strength from one observation** (FRT0000A091, lower Mount
  Sharp), not confirmed mineral maps. The clay signal in that cube is weak. Orbital hydrous
  detections are given only to 0.1°.
- **AI sand risk exists only where Curiosity took a test-set image** (319 images, 89 stops). Markers
  sit at the rover's position, and the terrain seen is within ~30 m of it. 32 positions are
  interpolated by drive number, so they are approximate. Routes more than 200 m from any analysed
  photo get "unknown" for sand.
- **The image route is a camera-view preview** (lower 60% of the frame, where labels exist), not
  distances in metres.
- **Walk time** is a planning estimate (Tobler's function rescaled to 3 km/h), not a validated EVA
  model.

## How to run

Needs Python 3.11 and about 2 GB of disk; no GPU.

```bash
git clone https://github.com/10apashmeenkaur-boop/space-apps.git && cd space-apps
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # CPU-only PyTorch, ~5 min
python app.py                                         # open http://127.0.0.1:7860
```

The first start downloads the DINOv2-small weights (~90 MB) from Hugging Face; after that the
app works offline except for the map tiles (NASA Mars Trek). Try **★ Example route** on the map,
click a coloured rover-photo marker, and pick a sample in the **AI Ground Check** tab.

The repo ships everything the app needs (`models/`, `geo/`, `samples/`, `results/`). To rebuild
from scratch (about 45 min on a laptop CPU; heavy files go to `SANDTRAP_DATA`, default `data/`):

```bash
python src/data.py          # stream AI4Mars, clean masks, previews, class counts
python src/features.py      # DINOv2 features (cached, resumable)
python src/train.py         # LogReg vs LightGBM, chosen by val mIoU
python src/evaluate.py      # test metrics, baseline, confusion matrix, examples
python src/make_samples.py  # 8 demo samples for the AI tab
python src/geo.py           # Gale layers: HiRISE slope, CRISM, traverse (Mars Trek, PDS, MMGIS)
python src/geolocate.py     # place AI4Mars test images by site/drive + AI results
pytest -q                   # planner, data and feature tests
```


## Extra experiment: an "unfamiliar terrain" check (tested, not yet in the app)

Mars is mostly unexplored, so a model can meet ground unlike anything it learned from. We tested
a simple novelty check: kNN distance (k=10) from each DINOv2 patch to 10,000 training patches,
with the threshold set at the 99th percentile of held-out patches (`src/novelty.py`,
`results/novelty.json`, `results/novelty_experiments.png`).

| Experiment | Result |
|---|---|
| Big rocks hidden from the reference, then shown | **AUROC 0.964**; flags 68.9% of big-rock pixels and 1.6% of other terrain ✅ |
| 26 Perseverance Navcam images (Jezero) vs Curiosity test images | **Negative result**: near field 15.5% flagged vs 25.7% for Curiosity ❌ |

So it spots terrain types it has not seen, but it does not separate a new crater as a whole. It
mostly reacts to rover hardware, shadows and sky. We report this as is.

## What's next (planned, not built)

- **"What you'll see":** the nearest real rover photo for each leg of a route, before you get there.
- **Helmet-camera ground check:** the same AI on live video at boot level.
- **Route modes:** safest, fastest, science-rich, **emergency return when low on oxygen** (no climbing), flattest.
- **Suit time + return-to-base:** a turn-back point and the safest path home.
- **Radiation dose per walk:** walk time × 0.67 mSv/day, the surface average measured by Curiosity's RAD.
- **Unfamiliar-terrain flags** in the app ("area of interest, not studied up close").
- **Mission-ready:** preload the maps for offline use, retrain on astronaut camera images, run on space-rated hardware.

## Tech stack

| Part | Tools |
|---|---|
| Full app (local) | Python 3.11, **Gradio** (tabs, upload, results), **Leaflet** map inside Gradio, HTML/CSS/JS |
| Web version (`web/`, free hosting) | Plain HTML + JavaScript + Leaflet; `web/marswalk.js` is a line-by-line port of the Python route engine; data precomputed by `src/build_static.py` |
| AI | **DINOv2-small** (Meta, frozen features) + **LightGBM** (trained by us), PyTorch (CPU), Hugging Face Transformers, scikit-learn |
| Planning | A* path planner on a probability cost map (rover photos); great-circle route sampling on HiRISE / MOLA grids (map); Tobler walking-speed rule |
| Geodata | NumPy, Pillow, OpenCV; tiles in the Mars 2000 geographic grid (EPSG:104905) |
| Testing | pytest (17 tests), Node.js for the JS-vs-Python check, a Python audit hook for runtime files |
| Hosting | Vercel (static, free) for `web/`; the full Gradio app runs locally |

## Data sources and how we verified them

| Data | Where it comes from | How we checked it |
|---|---|---|
| AI4Mars labels + Curiosity Navcam photos | NASA/JPL-Caltech, via Hugging Face `hassanjbara/AI4MARS` (CC0-1.0) | Mask values asserted ⊆ {0,1,2,3,255}; rover and >30 m pixels masked; previews checked by eye; every image hashed so no test image is in training |
| CTX global mosaic (~6 m) | NASA Mars Trek WMTS → Esri OnMars CTX service | `WMTSCapabilities.xml` read for the tile grid; a real tile over Gale downloaded and viewed |
| MOLA colour hillshade (463 m) | NASA Mars Trek WMTS | Same: capabilities read, Gale tile downloaded and viewed (crater + Mount Sharp visible) |
| HiRISE 1 m Gale DEM | NASA Mars Trek ImageServer `Gale_DEM_SMG_1m` (catalogue: "MRO HiRISE, Gale Crater DEM") | Extent and pixel size read from the service; overview hillshade checked by eye; 30 tiles downloaded with retries |
| MOLA MEGDR elevation (fallback) | PDS Geosciences Node, `megt00n090hb.img` | PDS label read (128 px/deg, big-endian int16); only the Gale rows range-read; elevations −4,648 to +1,412 m |
| CRISM FRT0000A091 (clay / sulfate) | PDS Geosciences Node, MTRDR summary parameters | PDS label read (band order, map projection); value ranges printed; labelled as "strength", not a mineral map |
| CRISM + OMEGA hydrous detections | NASA Mars Trek map service | Queried for the Gale window: 7 points (2 CRISM, 5 OMEGA), 0.1° precision |
| Curiosity traverse, 1,389 waypoints, science sites | NASA MMGIS (`mars.nasa.gov/mmgis-maps/MSL/...`) | Our filename decoding checked against it: site/drive 55_940 → sol 1384 (lat −4.682396, lon 137.357345); 76_568 → sol 2434; 73_2958 → sol 2311, all exact |
| Perseverance Navcam (novelty test only) | NASA raw-image JSON feed `mars.nasa.gov/rss/api/?feed=raw_images&category=mars2020...` | Image URLs saved in `results/perseverance_images.json` |

**Facts quoted in the presentation, checked against primary sources:**
- Spirit "became stuck in soft soil" on 1 May 2009 (NASA Science, Spirit mission page).
- One-way Earth–Mars signal "anywhere from 4 minutes to 22 minutes" (JPL "On a Mission" podcast, rover driver Vandi Verma).
- Curiosity RAD measured 0.67 mSv/day on the surface, Aug 2012–Jun 2013 (SwRI press release, RAD team).
- DeepLabv3 "over 96% overall classification accuracy" (AI4Mars paper, CVPR 2021 Workshops).
- SPOCv2 "Total Accuracy: 91.41%" (Zooniverse AI4Mars results page).

## Tests and checks we ran

**Automated tests (`pytest -q`): 17/17 pass.**

| Test file | What it checks |
|---|---|
| `tests/test_planner.py` (5) | Open field gives a straight route; a rock wall forces a detour; a fully blocked field returns "no route" and HOLD; a goal outside the near field is refused; sand raises cost and is reported on the right side |
| `tests/test_data.py` (6) | RGB masks use channel 0, and rover / far-range pixels are masked; bad label values are rejected; image ids are content hashes; the cell majority-label and ≥60% rule; class-balanced sampling keeps rare big rock; probabilities upsample to 1024×1024 |
| `tests/test_geo.py` (5) | Navcam filename parsing (`F0600888` → site 60, drive 888); exact and interpolated waypoint lookup; a HiRISE-only route (distance, legs, AI photos sorted by distance, Kimberley found); a route leaving HiRISE falls back to MOLA and is labelled "mixed"; a route outside Gale is refused |
| `tests/test_web_parity.py` (1) | The browser version (`web/marswalk.js`, run in Node) gives the same distance, climb, slope shares, risk, AI photo count, stop count and leg sources as the Python version, on two routes |

**Other checks:**
- **No data leakage:** pixel-hash ids; asserted that no test image is in train or validation.
- **Official test set only:** final numbers come only from AI4Mars `test_min1/2/3` (322 expert-labelled images), with a majority-class baseline.
- **Model choice by validation, not test:** LightGBM val mIoU 0.746 vs Logistic Regression 0.697.
- **Clean-clone run:** fresh `git clone`, no `data/` folder. Every code path ran; the server answered HTTP 200 and ran an AI check in 2.3 s.
- **Browser check of the web version:** example route report identical to the app; 97 map markers; 8 AI result cards and the metrics table load; no console errors.
- **Speed:** ~0.5 s of model time per image on a laptop CPU (i5-12450H); ~2 s end to end.

## Repo layout

```
app.py              Gradio app (Marswalk Map, AI Ground Check, Data & limits)
config.py           constants (IMG=448, GRID=32, classes, seeds, planner weights)
src/data.py         streaming loader, mask cleaning, content-hash ids
src/features.py     DINOv2 feature extractor (batch, cache, resume)
src/train.py        LogReg + LightGBM, by-image validation
src/evaluate.py     pixel metrics, baseline, confusion matrix, examples
src/planner.py      image cost map + 8-direction A*
src/briefing.py     drive briefing text
src/geo.py          builds geo/ layers from NASA sources
src/geolocate.py    site/drive from Navcam filenames -> positions + AI results
src/marswalk.py     map route analysis (HiRISE slope, AI sand, stops, proven ground, walk time)
src/novelty.py      unfamiliar-terrain (kNN novelty) check + experiments
src/build_static.py builds the static web version into web/
web/                static web version (deploy this folder)
docs/               presentation (PPTX + PDF)
tests/              pytest
```

## What the app reads at runtime (deployment check)

Checked with a Python audit hook while exercising every code path (map layers, marker popups,
HiRISE-only and HiRISE+MOLA routes, AI ground check on all samples), run from a fresh `git clone`
with **no `data/` folder** and `SANDTRAP_DATA` pointing at a folder that does not exist:

| Path | Used for |
|---|---|
| `app.py`, `config.py`, `src/{briefing,features,geo,infer,marswalk,planner}.py` | code |
| `models/classifier.joblib` | LightGBM terrain classifier |
| `results/metrics.json`, `results/confusion_matrix.png` | accuracy table and figure |
| `samples/*.png` (8) | AI Ground Check examples |
| `geo/layers.json` | overlays, traverse, waypoints, science stops |
| `geo/hirise_dem20.npz`, `geo/mola_dem.npz` | slope and elevation for routes |
| `geo/ai_images.json`, `geo/ai_thumbs/*.jpg` (319) | geolocated AI markers and popups |
| `facebook/dinov2-small` | downloaded from the Hugging Face Hub on first start |

Nothing is read from `data/`, which holds only build-time caches (dataset images, features, DEM
tiles). It is excluded from git. The clean-copy server answered
HTTP 200 and ran an AI ground check through the Gradio API in 2.3 s.

## Deploy the web version (Vercel, free)

1. Go to vercel.com → **Add New… → Project** → import the GitHub repo `space-apps`.
2. Set **Root Directory** to `web`, **Framework Preset** to *Other*, and leave the build command empty.
3. Press **Deploy**. Ours is live at https://marswalk-planner-jet.vercel.app

To rebuild `web/` after changing data: `python src/build_static.py`.

## Licences

| Item | Licence / terms |
|---|---|
| This code | **Apache-2.0** (`LICENSE`) |
| AI4Mars dataset (NASA/JPL-Caltech, via `hassanjbara/AI4MARS`) | **CC0-1.0** |
| DINOv2-small model weights (`facebook/dinov2-small`, Meta) | **Apache-2.0** (per its Hugging Face model card) |
| CTX global mosaic beta01 | free use for all purposes; credit NASA/JPL/MSSS/The Murray Lab (per the service description) |
| MOLA, HiRISE, CRISM, OMEGA products, MMGIS traverse | NASA / ESA mission data, credited below |

## Credits

Data: NASA/JPL-Caltech **AI4Mars**; **NASA Mars Trek** (CTX mosaic: NASA/JPL/MSSS/The Murray Lab; MOLA: NASA/JPL/GSFC; HiRISE Gale DEM: NASA/JPL/University of Arizona; CRISM/OMEGA hydrous
detections); **PDS Geosciences Node** (MOLA MEGDR, CRISM MTRDR); **NASA MMGIS** (Curiosity traverse, waypoints,
science sites).
Model: DINOv2 (Meta AI). Libraries: PyTorch, Transformers, scikit-learn, LightGBM, Gradio, Leaflet.
