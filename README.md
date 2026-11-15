# Marswalk Planner: Gale Crater, with an AI ground-check layer

**NASA Space Apps 2026, Challenge #10: Interplanetary Survival Guide: Martian Map.**
The challenge asks for *a layered, integrated view of a location or route on the Martian surface
that pulls together data from multiple NASA science missions and could help a human explorer plan
and carry out a successful Marswalk while doing new and exciting science along the way.*

Demo video (30 s): `<add link>` · Run it yourself in about 5 minutes on any laptop (no GPU): see [How to run](#how-to-run).

**AI ground check, official AI4Mars test set (`test_min3`): mIoU 0.777 vs 0.056 for a majority-class
baseline; sand recall 96.3% vs 0%.** Runs on a laptop CPU (~0.5 s per image).

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
git clone <this repo's URL> marswalk-planner && cd marswalk-planner
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
