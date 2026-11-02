"""Feature construction for LinUCB and the learned replay trigger."""
from __future__ import annotations

import math

import numpy as np

from cca.types import Context

DTYPES = ["plan", "tool_choice", "code_edit"]


def onehot_d(d: str) -> list[float]:
    return [1.0 if d == k else 0.0 for k in DTYPES]


def proposal_vec(ctx: Context, i: int, use_dtype: bool = True) -> np.ndarray:
    p = ctx.proposals[i]
    x = [1.0, p.self_confidence, math.log1p(len(p.justification)) / 5.0,
         math.log1p(len(str(p.action.args))) / 5.0]
    if use_dtype:
        x += onehot_d(ctx.decision_type)
    x += list(ctx.task_features)
    return np.asarray(x, dtype=float)


def judge_uncertainty(ctx: Context) -> float:
    """1 - margin between the top two judge scores over distinct groups (0 if unknown)."""
    if not ctx.judge_scores:
        return 0.0
    best: dict[int, float] = {}
    for g, s in zip(ctx.groups, ctx.judge_scores):
        best[g] = max(best.get(g, -1e9), s)
    v = sorted(best.values(), reverse=True)
    return 1.0 if len(v) < 2 else float(max(0.0, min(1.0, 1.0 - (v[0] - v[1]))))


def disagreement_frac(ctx: Context) -> float:
    n = len(ctx.proposals)
    return 0.0 if n < 2 else (ctx.n_groups - 1) / (n - 1)


def trigger_vec(ctx: Context) -> np.ndarray:
    n = max(1, len(ctx.proposals))
    return np.asarray(
        [1.0, disagreement_frac(ctx), judge_uncertainty(ctx),
         sum(p.self_confidence for p in ctx.proposals) / n,
         sum(math.log1p(len(p.justification)) for p in ctx.proposals) / (5.0 * n),
         ctx.step / max(1, ctx.max_steps)] + onehot_d(ctx.decision_type), dtype=float)
