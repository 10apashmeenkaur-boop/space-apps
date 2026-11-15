"""Template drive briefing from the terrain probabilities and the planned route."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.planner import ROCK, SAND, near_field_top  # noqa: E402


def _side(cols, n):
    """Where a set of columns sits: left / centre / right."""
    if len(cols) == 0:
        return None
    m = np.mean(cols) / n
    return "left" if m < 0.4 else "right" if m > 0.6 else "centre"


def verdict(risk):
    if risk is None:
        return "HOLD"
    if risk < 0.15:
        return "GO"
    if risk < 0.35:
        return "GO WITH CAUTION"
    return "HOLD"


def make_briefing(result):
    p, path = result["prob"], result["path"]
    n = p.shape[0]
    nf = p[near_field_top(n):]
    lab = nf.argmax(-1)
    frac = {c: float((lab == k).mean()) for k, c in enumerate(C.CLASSES)}
    sand_cols = np.nonzero(lab == SAND)[1]
    rock_cells = np.argwhere(nf[..., ROCK] > C.ROCK_BLOCK_P)
    low_conf = float((nf.max(-1) < 0.5).mean())

    lines = []
    s = f"Sand covers {frac['sand']:.0%} of the near field"
    side = _side(sand_cols, n)
    lines.append(s + (f", mostly {side}." if frac["sand"] >= 0.02 and side else "."))
    if len(rock_cells):
        lines.append(f"Big rock detected {_side(rock_cells[:, 1], n)} ({len(rock_cells)} blocked cells).")
    else:
        lines.append("No big rocks detected in the near field.")
    lines.append(f"Ground mix: soil {frac['soil']:.0%}, bedrock {frac['bedrock']:.0%}.")
    if low_conf > 0.2:
        lines.append(f"Model is unsure about {low_conf:.0%} of the near field - treat with care.")

    if path is None:
        lines.append("No safe route to the target - re-image from another angle.")
    else:
        dc = np.array([c for _, c in path]) - result["start"][1]
        bend = dc[np.argmax(np.abs(dc))] if len(dc) else 0
        shape = "runs straight" if abs(bend) <= 2 else ("curves right" if bend > 0 else "curves left")
        lines.append(f"Safe route {shape}, length ~{len(path)} grid cells, risk score {result['risk']:.2f}.")

    v = verdict(result["risk"])
    return {"verdict": v, "text": " ".join(lines), "lines": lines, "near_field_fractions": frac,
            "risk": result["risk"]}
