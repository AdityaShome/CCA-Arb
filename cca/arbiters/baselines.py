"""Baseline arbiters (GUIDE 6.7). DRF / RR / BDA are INTERPRETATIONS, see docs/baselines.md."""
from __future__ import annotations

import math

from cca.arbiters.base import Arbiter, group_support
from cca.types import Context


def _pick_group(arb: Arbiter, ctx: Context, support: dict[int, float],
                score_fn=None) -> int:
    """Pick best group by support; return index of its best-ordered proposal."""
    gbest = max(support, key=lambda g: (support[g], -min(
        arb.order(ctx.proposals[i].agent_id) for i, gg in enumerate(ctx.groups) if gg == g)))
    idx = [i for i, g in enumerate(ctx.groups) if g == gbest]
    return min(idx, key=lambda i: arb.order(ctx.proposals[i].agent_id))


class Majority(Arbiter):
    name = "majority"

    def select(self, ctx):
        return _pick_group(self, ctx, group_support(ctx))


class AcceptanceRep(Arbiter):
    """Judge-driven reputation: +1 when accepted, geometric decay. Judge sees justifications."""
    name = "acceptance_rep"
    needs_judge = True

    def __init__(self, agent_order, seed=0, beta=1.0, decay=0.98, **kw):
        super().__init__(agent_order, seed)
        self.beta, self.decay = beta, decay
        self.rep = {a: 0.0 for a in agent_order}

    def select(self, ctx):
        assert ctx.judge_scores is not None, "acceptance_rep needs a judge"
        scores = [ctx.judge_scores[i] + self.beta * (1 - self.decay) * self.rep.get(p.agent_id, 0.0)
                  for i, p in enumerate(ctx.proposals)]
        return self.best_by(ctx, scores)

    def observe_accept(self, ctx, chosen):
        for a in self.rep:
            self.rep[a] *= self.decay
        a = ctx.proposals[chosen].agent_id
        self.rep[a] = self.rep.get(a, 0.0) + 1.0


class DRF(Arbiter):
    """Reputation filtering + reputation*confidence weighted vote (interpretation)."""
    name = "drf"

    def __init__(self, agent_order, seed=0, eta=0.1, theta=0.3, **kw):
        super().__init__(agent_order, seed)
        self.eta, self.theta = eta, theta
        self.rep = {a: 0.5 for a in agent_order}

    def select(self, ctx):
        keep = [p.agent_id for p in ctx.proposals if self.rep.get(p.agent_id, 0.5) >= self.theta]
        if not keep:
            keep = [p.agent_id for p in ctx.proposals]
        w = {p.agent_id: (self.rep.get(p.agent_id, 0.5) * max(p.self_confidence, 1e-3)
                          if p.agent_id in keep else 0.0) for p in ctx.proposals}
        return _pick_group(self, ctx, group_support(ctx, w))

    def observe_accept(self, ctx, chosen):
        # peer evaluation: fraction of OTHER agents backing my group (no ground truth used)
        n = len(ctx.proposals)
        for p, g in zip(ctx.proposals, ctx.groups):
            peers = [gg for q, gg in zip(ctx.proposals, ctx.groups) if q is not p]
            peer = sum(gg == g for gg in peers) / max(1, len(peers)) if n > 1 else 1.0
            a = p.agent_id
            self.rep[a] = (1 - self.eta) * self.rep.get(a, 0.5) + self.eta * peer


class RR(Arbiter):
    """Remember & Reweight: persistent multiplicative weights driven by agreement with the accepted action."""
    name = "rr"

    def __init__(self, agent_order, seed=0, eta=0.2, **kw):
        super().__init__(agent_order, seed)
        self.eta = eta
        self.w = {a: 1.0 for a in agent_order}

    def select(self, ctx):
        return _pick_group(self, ctx, group_support(ctx, self.w))

    def observe_accept(self, ctx, chosen):
        gc = ctx.groups[chosen]
        for p, g in zip(ctx.proposals, ctx.groups):
            self.w[p.agent_id] = self.w.get(p.agent_id, 1.0) * math.exp(self.eta if g == gc else -self.eta)
        z = sum(self.w.values()) / len(self.w)
        for a in self.w:
            self.w[a] /= z


class BDA(Arbiter):
    """Bayesian Dialectical Argumentation (interpretation): posterior over which group is correct."""
    name = "bda"
    needs_judge = True

    def __init__(self, agent_order, seed=0, gamma=1.0, **kw):
        super().__init__(agent_order, seed)
        self.gamma = gamma
        self.ab = {a: [1.0, 1.0] for a in agent_order}

    def _r(self, a):
        x, y = self.ab.setdefault(a, [1.0, 1.0])
        return min(max(x / (x + y), 1e-3), 1 - 1e-3)

    def select(self, ctx):
        assert ctx.judge_scores is not None, "bda needs a judge"
        logp: dict[int, float] = {}
        for g in set(ctx.groups):
            lp = 0.0
            for p, gg in zip(ctx.proposals, ctx.groups):
                r = self._r(p.agent_id)
                lp += math.log(r) if gg == g else math.log(1 - r)
            js = [s for s, gg in zip(ctx.judge_scores, ctx.groups) if gg == g]
            lp += self.gamma * math.log(max(sum(js) / len(js), 1e-3))
            logp[g] = lp
        return _pick_group(self, ctx, logp_to_support(logp))

    def observe_accept(self, ctx, chosen):
        gc = ctx.groups[chosen]
        for p, g in zip(ctx.proposals, ctx.groups):
            self.ab.setdefault(p.agent_id, [1.0, 1.0])[0 if g == gc else 1] += 1.0


def logp_to_support(logp: dict[int, float]) -> dict[int, float]:
    m = max(logp.values())
    return {g: math.exp(v - m) for g, v in logp.items()}


class Oracle(Arbiter):
    """Upper bound: picks the proposal with highest replayed Y. Falls back to majority when no replay."""
    name = "oracle"
    uses_oracle = True

    def select(self, ctx):
        if ctx.oracle_outcomes is None:
            return _pick_group(self, ctx, group_support(ctx))
        return self.best_by(ctx, ctx.oracle_outcomes)
