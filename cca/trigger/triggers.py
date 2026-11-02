"""Replay triggers (GUIDE 6.8). Called only at disagreements."""
from __future__ import annotations

import random

import numpy as np

from cca.features import disagreement_frac, judge_uncertainty, trigger_vec
from cca.types import Context


class Trigger:
    name = "base"
    needs_judge = False

    def __init__(self, seed: int = 0, **_):
        self.rng = random.Random(seed)

    def should_replay(self, ctx: Context) -> bool:
        raise NotImplementedError

    def observe(self, ctx: Context, ys: list[float]) -> None:
        """Called after a replay with the outcomes (for learned triggers)."""


class Always(Trigger):
    name = "always"

    def should_replay(self, ctx):
        return True


class Never(Trigger):
    name = "never"

    def should_replay(self, ctx):
        return False


class RandomP(Trigger):
    name = "random_p"

    def __init__(self, seed=0, p=0.3, **kw):
        super().__init__(seed)
        self.p = p

    def should_replay(self, ctx):
        return self.rng.random() < self.p


class Threshold(Trigger):
    """Replay iff g = a*disagreement_frac + b*judge_uncertainty exceeds tau."""
    name = "threshold"
    needs_judge = True

    def __init__(self, seed=0, tau=0.5, a=0.5, b=0.5, **kw):
        super().__init__(seed)
        self.tau, self.a, self.b = tau, a, b

    def score(self, ctx):
        return self.a * disagreement_frac(ctx) + self.b * judge_uncertainty(ctx)

    def should_replay(self, ctx):
        return self.score(ctx) > self.tau


class Learned(Trigger):
    """Logistic regression predicting max|delta| > eps from trigger features.

    Replays always during warm-up, then replays iff P(informative) > tau.
    """
    name = "learned"
    needs_judge = True

    def __init__(self, seed=0, eps=0.2, tau=0.4, warmup=30, retrain_every=10,
                 l2=1e-2, lr=0.5, iters=1000, explore=0.05, **kw):
        super().__init__(seed)
        self.eps, self.tau, self.warmup, self.every = eps, tau, warmup, retrain_every
        self.l2, self.lr, self.iters = l2, lr, iters
        self.explore = explore  # keeps labelling skipped regions (selection-bias guard)
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.w: np.ndarray | None = None

    def prob(self, ctx) -> float:
        if self.w is None:
            return 1.0
        z = float(trigger_vec(ctx) @ self.w)
        return 1.0 / (1.0 + np.exp(-z))

    def should_replay(self, ctx):
        if len(self.y) < self.warmup or self.rng.random() < self.explore:
            return True
        return self.prob(ctx) > self.tau

    def observe(self, ctx, ys):
        m = sum(ys) / len(ys)
        self.X.append(trigger_vec(ctx))
        self.y.append(1.0 if max(abs(v - m) for v in ys) > self.eps else 0.0)
        if len(self.y) >= self.warmup and len(self.y) % self.every == 0:
            self._fit()

    def _fit(self):
        X, y = np.stack(self.X), np.asarray(self.y)
        w = np.zeros(X.shape[1])
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-(X @ w)))
            w -= self.lr * (X.T @ (p - y) / len(y) + self.l2 * w)
        self.w = w


REGISTRY = {c.name: c for c in (Always, Never, RandomP, Threshold, Learned)}


def build_trigger(name: str, seed: int = 0, **params) -> Trigger:
    return REGISTRY[name](seed=seed, **params)
