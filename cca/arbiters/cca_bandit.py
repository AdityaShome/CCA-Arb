"""CCA arbiters (GUIDE 6.6): bandit over per-(agent, decision-type) reliability.

v1 `cca_ts`   Beta-Bernoulli Thompson sampling.
v2 `cca_linucb` LinUCB over context features.

Replayed step: every evaluated agent is updated with its own outcome Y ("y" credit) or
with the paired credit 0.5 + (Y_i - mean_j Y_j) clipped to [0,1] ("delta" credit).
Non-replayed step: only the accepted agent, with the delayed episode reward, at weight w.
"""
from __future__ import annotations

import numpy as np

from cca.arbiters.base import Arbiter
from cca.features import proposal_vec
from cca.types import Context


def _credit(ys: list[float], mode: str) -> list[float]:
    if mode == "y":
        return list(ys)
    m = sum(ys) / len(ys)
    return [float(min(1.0, max(0.0, 0.5 + (y - m)))) for y in ys]


class CCAThompson(Arbiter):
    name = "cca_ts"

    def __init__(self, agent_order, seed=0, use_dtype=True, w_delayed=0.25, gamma=1.0,
                 credit="y", prior=1.0, **kw):
        super().__init__(agent_order, seed)
        self.use_dtype, self.w, self.gamma = use_dtype, w_delayed, gamma
        self.credit, self.prior = credit, prior
        self.ab: dict[tuple, list[float]] = {}
        self.pending: list[tuple] = []
        self.np_rng = np.random.default_rng(seed)

    def key(self, agent, d):
        return (agent, d if self.use_dtype else "all")

    def _get(self, k):
        return self.ab.setdefault(k, [self.prior, self.prior])

    def select(self, ctx):
        scores = [float(self.np_rng.beta(*self._get(self.key(p.agent_id, ctx.decision_type))))
                  for p in ctx.proposals]
        return self.best_by(ctx, scores)

    def _update(self, k, y, weight):
        ab = self._get(k)
        if self.gamma < 1.0:  # shrink toward prior (non-stationarity)
            ab[0] = self.prior + self.gamma * (ab[0] - self.prior)
            ab[1] = self.prior + self.gamma * (ab[1] - self.prior)
        ab[0] += weight * y
        ab[1] += weight * (1 - y)

    def observe_replay(self, ctx, chosen, ys):
        seen = {}
        for p, c in zip(ctx.proposals, _credit(ys, self.credit)):
            seen[self.key(p.agent_id, ctx.decision_type)] = c
        for k, c in seen.items():
            self._update(k, c, 1.0)

    def queue_delayed(self, ctx, chosen):
        self.pending.append(self.key(ctx.proposals[chosen].agent_id, ctx.decision_type))

    def flush_episode(self, reward):
        for k in self.pending:
            self._update(k, float(reward), self.w)
        self.pending = []

    def reliability(self) -> dict[tuple, float]:
        return {k: a / (a + b) for k, (a, b) in self.ab.items()}


class CCALinUCB(Arbiter):
    name = "cca_linucb"

    def __init__(self, agent_order, seed=0, use_dtype=True, alpha=1.0, lam=1.0,
                 w_delayed=0.25, credit="y", **kw):
        super().__init__(agent_order, seed)
        self.use_dtype, self.alpha, self.lam, self.w, self.credit = use_dtype, alpha, lam, w_delayed, credit
        self.A: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.pending: list[tuple[str, np.ndarray]] = []

    def _init(self, a, dim):
        if a not in self.A:
            self.A[a], self.b[a] = self.lam * np.eye(dim), np.zeros(dim)

    def select(self, ctx):
        scores = []
        for i, p in enumerate(ctx.proposals):
            x = proposal_vec(ctx, i, self.use_dtype)
            self._init(p.agent_id, len(x))
            Ai = np.linalg.inv(self.A[p.agent_id])
            theta = Ai @ self.b[p.agent_id]
            scores.append(float(theta @ x + self.alpha * np.sqrt(max(x @ Ai @ x, 0.0))))
        return self.best_by(ctx, scores)

    def _update(self, a, x, y, weight):
        self._init(a, len(x))
        self.A[a] += weight * np.outer(x, x)
        self.b[a] += weight * y * x

    def observe_replay(self, ctx, chosen, ys):
        for i, (p, c) in enumerate(zip(ctx.proposals, _credit(ys, self.credit))):
            self._update(p.agent_id, proposal_vec(ctx, i, self.use_dtype), c, 1.0)

    def queue_delayed(self, ctx, chosen):
        self.pending.append((ctx.proposals[chosen].agent_id,
                             proposal_vec(ctx, chosen, self.use_dtype)))

    def flush_episode(self, reward):
        for a, x in self.pending:
            self._update(a, x, float(reward), self.w)
        self.pending = []
