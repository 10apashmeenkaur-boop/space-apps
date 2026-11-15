import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.data import clean_mask, image_id  # noqa: E402
from src.features import balanced_pick, cell_labels  # noqa: E402
from src.infer import pixel_pred  # noqa: E402


def test_clean_mask_rgb_takes_channel_0_and_applies_rover_range():
    m = np.zeros((8, 8, 3), np.uint8)
    m[..., 0] = 2
    rover = np.zeros((8, 8), np.uint8)
    rover[0, 0] = 1
    rng = np.zeros((8, 8), np.uint8)
    rng[7, 7] = 1
    out = clean_mask(Image.fromarray(m), Image.fromarray(rover), Image.fromarray(rng))
    assert out.shape == (8, 8)
    assert out[0, 0] == C.IGNORE and out[7, 7] == C.IGNORE and out[3, 3] == 2


def test_clean_mask_rejects_bad_values():
    with pytest.raises(AssertionError):
        clean_mask(Image.fromarray(np.full((4, 4), 7, np.uint8)))


def test_image_id_is_content_hash():
    a = np.zeros((4, 4), np.uint8)
    b = a.copy()
    b[0, 0] = 1
    assert image_id(a) == image_id(a.copy()) != image_id(b)


def test_cell_labels_majority_and_60pct_rule():
    m = np.full((C.ORIG, C.ORIG), C.IGNORE, np.uint8)
    m[: C.CELL, : C.CELL] = 2                      # cell 0: all sand -> valid, sand
    m[: C.CELL, C.CELL: C.CELL + C.CELL // 2] = 1  # cell 1: 50% labelled -> invalid
    m[: C.CELL, 2 * C.CELL: 3 * C.CELL] = 0         # cell 2: 3/4 soil, 1/4 bedrock -> soil
    m[: C.CELL // 4, 2 * C.CELL: 3 * C.CELL] = 1
    lab, valid = cell_labels(m)
    assert valid[0] and lab[0] == 2
    assert not valid[1]
    assert valid[2] and lab[2] == 0
    assert valid.sum() == 2


def test_balanced_pick_caps_and_keeps_rare_class():
    labels = np.zeros(1024, int)
    labels[:5] = 3  # 5 big-rock cells among 1019 soil
    picked = balanced_pick(labels, np.ones(1024, bool), 250, np.random.default_rng(0))
    assert len(picked) == 250 and len(set(picked)) == 250
    assert (labels[picked] == 3).sum() == 5


def test_pixel_pred_upsamples_to_full_size():
    prob = np.zeros((C.GRID, C.GRID, C.N_CLASSES), np.float32)
    prob[..., 1] = 1
    pred = pixel_pred(prob)
    assert pred.shape == (C.ORIG, C.ORIG) and (pred == 1).all()
