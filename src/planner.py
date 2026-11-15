"""Phase 4: cost map + A* route preview in image space.

The camera view is not a top-down map, so this is an *image-space route preview*
for operators, limited to the near field (lower 60% of the image) where labels
exist and the rover can actually reach.
"""
import heapq
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

SAND, ROCK = C.CLASSES.index("sand"), C.CLASSES.index("bigrock")
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def to_plan_grid(prob, n=C.PLAN_GRID):
    """(g,g,4) probabilities -> (n,n,4) planning grid."""
    if prob.shape[0] == n:
        return prob.astype(np.float32)
    interp = cv2.INTER_AREA if prob.shape[0] > n else cv2.INTER_LINEAR
    return cv2.resize(prob.astype(np.float32), (n, n), interpolation=interp)


def near_field_top(n=C.PLAN_GRID):
    return int(math.floor(n * (1 - C.NEAR_FIELD_FRAC)))


def cost_map(prob):
    """cost = 1 + 20 P(sand) + 50 P(big rock) + 5 (1 - max prob); P(big rock) > 0.5 is blocked.

    Rows above the near field are blocked (out of reach / unlabelled range).
    """
    cost = 1 + C.W_SAND * prob[..., SAND] + C.W_ROCK * prob[..., ROCK] + C.W_UNCERT * (1 - prob.max(-1))
    blocked = prob[..., ROCK] > C.ROCK_BLOCK_P
    blocked[: near_field_top(prob.shape[0])] = True
    return cost.astype(np.float32), blocked


def astar(cost, blocked, start, goal):
    """8-connected A*. Step cost = step length x mean cell cost. Returns [(r,c),...] or None."""
    H, W = cost.shape
    if blocked[start] or blocked[goal]:
        return None
    floor = float(cost[~blocked].min()) if (~blocked).any() else 1.0

    def h(p):
        dr, dc = abs(p[0] - goal[0]), abs(p[1] - goal[1])
        return floor * (max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc))

    g = {start: 0.0}
    parent = {start: None}
    heap = [(h(start), 0.0, start)]
    closed = set()
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        closed.add(cur)
        for dr, dc in MOVES:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < H and 0 <= nb[1] < W) or blocked[nb] or nb in closed:
                continue
            if dr and dc and (blocked[cur[0] + dr, cur[1]] or blocked[cur[0], cur[1] + dc]):
                continue  # no corner cutting past an obstacle
            step = (math.sqrt(2) if dr and dc else 1.0) * 0.5 * (cost[cur] + cost[nb])
            ng = gc + step
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(heap, (ng + h(nb), ng, nb))
    return None


def default_goal(blocked, n=C.PLAN_GRID):
    """Farthest reachable-looking point straight ahead: centre column, top of the near field,
    stepping down (and sideways) until an unblocked cell is found."""
    top, mid = near_field_top(n), n // 2
    for r in range(top, n - 1):
        for off in [0, -1, 1, -2, 2, -3, 3, -4, 4]:
            c = mid + off
            if 0 <= c < n and not blocked[r, c]:
                return (r, c)
    return (top, mid)


def plan(prob, goal=None):
    """prob: (g,g,4) probabilities. goal: (row, col) on the planning grid or None."""
    p = to_plan_grid(prob)
    n = p.shape[0]
    cost, blocked = cost_map(p)
    start = (n - 1, n // 2)
    blocked[start] = False  # the rover is standing here
    if goal is None:
        goal = default_goal(blocked, n)
    goal = (int(np.clip(goal[0], near_field_top(n), n - 1)), int(np.clip(goal[1], 0, n - 1)))
    path = astar(cost, blocked, start, goal)
    risk = None
    if path:
        rr, cc = zip(*path)
        risk = float(np.mean(p[rr, cc, SAND] + p[rr, cc, ROCK]))
    return {"prob": p, "cost": cost, "blocked": blocked, "start": start, "goal": goal,
            "path": path, "risk": risk}
