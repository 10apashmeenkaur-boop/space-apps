"""Shared inference helpers: classifier -> class probabilities -> pixel map."""
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def load_classifier(path=None):
    return joblib.load(path or C.MODELS_DIR / "classifier.joblib")


def proba_grid(clf, grid_feats):
    """(32,32,384) features -> (32,32,4) class probabilities."""
    X = grid_feats.reshape(-1, C.FEAT_DIM).astype(np.float32)
    return clf.predict_proba(X).reshape(C.GRID, C.GRID, C.N_CLASSES).astype(np.float32)


def upsample(prob, size=C.ORIG):
    """Bilinear upsample (g,g,4) probabilities to (size,size,4)."""
    return cv2.resize(prob, (size, size), interpolation=cv2.INTER_LINEAR)


def pixel_pred(prob, size=C.ORIG):
    return upsample(prob, size).argmax(-1).astype(np.uint8)
