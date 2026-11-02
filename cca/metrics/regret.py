"""Regret / persuasion-drift helpers over decision records."""
from __future__ import annotations

import numpy as np


def cumulative_regret(decisions: list[dict]) -> np.ndarray:
    """Cumulative regret over REPLAYED decisions only (the only ones where Y is observed)."""
    r = [d["regret"] for d in decisions if d.get("replayed") and d.get("regret") is not None]
    return np.cumsum(r) if r else np.zeros(0)


def acceptance_share(decisions: list[dict], agent: str, window: int = 50) -> np.ndarray:
    """Rolling share of disagreements won by `agent` (x-axis: decision index)."""
    a = np.array([1.0 if d["chosen_agent"] == agent else 0.0 for d in decisions])
    if len(a) == 0:
        return a
    k = np.ones(min(window, len(a)))
    return np.convolve(a, k, mode="valid") / len(k)


def mean_true_y(decisions: list[dict], agent: str) -> float | None:
    """Mean replayed outcome of `agent`'s proposals (needs replayed decisions)."""
    ys = [d["ys"][d["agents"].index(agent)] for d in decisions
          if d.get("replayed") and agent in d["agents"]]
    return float(np.mean(ys)) if ys else None
