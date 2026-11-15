import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.briefing import make_briefing  # noqa: E402
from src.planner import ROCK, astar, cost_map, near_field_top, plan  # noqa: E402

N = C.PLAN_GRID
SOIL = C.CLASSES.index("soil")


def field(kind=SOIL):
    p = np.zeros((N, N, C.N_CLASSES), np.float32)
    p[..., kind] = 1.0
    return p


def test_empty_field_is_straight_line():
    r = plan(field(), goal=(near_field_top(), N // 2))
    assert r["path"] is not None
    assert all(c == N // 2 for _, c in r["path"])
    assert r["path"][0] == (N - 1, N // 2) and r["path"][-1] == (near_field_top(), N // 2)
    assert r["risk"] == 0.0


def test_rock_wall_goes_around():
    p = field()
    wall_row = N - 15
    p[wall_row, 10:54] = 0
    p[wall_row, 10:54, ROCK] = 1.0
    r = plan(p, goal=(near_field_top(), N // 2))
    assert r["path"] is not None
    assert all(not (rr == wall_row and 10 <= cc < 54) for rr, cc in r["path"])
    assert any(cc < 10 or cc >= 54 for _, cc in r["path"])


def test_fully_blocked_returns_none():
    p = field()
    p[N - 10, :] = 0
    p[N - 10, :, ROCK] = 1.0
    r = plan(p, goal=(near_field_top(), N // 2))
    assert r["path"] is None and r["risk"] is None
    b = make_briefing(r)
    assert b["verdict"] == "HOLD" and "re-image" in b["text"]


def test_astar_rejects_blocked_goal():
    cost, blocked = cost_map(field())
    assert astar(cost, blocked, (N - 1, 0), (0, 0)) is None  # row 0 is outside the near field


def test_sand_raises_cost():
    p = field()
    p[N - 20:, : N // 2] = 0
    p[N - 20:, : N // 2, C.CLASSES.index("sand")] = 1.0
    cost, _ = cost_map(p)
    assert cost[N - 1, 0] > cost[N - 1, N - 1]
    b = make_briefing(plan(p))
    assert "left" in b["lines"][0]
