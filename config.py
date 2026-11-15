"""Project-wide constants and paths."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Heavy files (raw images, feature caches) live outside the repo by default.
DATA_DIR = Path(os.environ.get("SANDTRAP_DATA", ROOT / "data"))
FEAT_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SAMPLES_DIR = ROOT / "samples"

HF_DATASET = "hassanjbara/AI4MARS"
TEST_SPLITS = ["test_min1", "test_min2", "test_min3"]
N_TRAIN_SHARDS = 13

DINO_MODEL = "facebook/dinov2-small"
IMG = 448            # model input size
PATCH = 14           # DINOv2 patch size
GRID = IMG // PATCH  # 32 x 32 patch grid
FEAT_DIM = 384
ORIG = 1024          # Navcam image size
CELL = ORIG // GRID  # 32 px of the original image per grid cell

CLASSES = ["soil", "bedrock", "sand", "bigrock"]
N_CLASSES = len(CLASSES)
IGNORE = 255

# ImageNet normalisation
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

SEED = 42

# Data selection
N_TRAIN_IMAGES = 1500
MIN_LABELLED_FRAC = 0.05   # skip train images with <5% labelled pixels
CELL_MIN_LABELLED = 0.60   # a grid cell needs >=60% labelled pixels to be used
MAX_CELLS_PER_IMAGE = 250
VAL_FRAC = 0.15

# Planner
PLAN_GRID = 64
NEAR_FIELD_FRAC = 0.60     # plan only in the lower 60% of the image
W_SAND = 20.0
W_ROCK = 50.0
W_UNCERT = 5.0
ROCK_BLOCK_P = 0.5

# Overlay colours (RGB)
COLORS = {
    "soil": (150, 95, 55),
    "bedrock": (140, 140, 150),
    "sand": (240, 200, 40),
    "bigrock": (220, 40, 40),
}
