"""Arbiter interface.

Per disagreement the episode loop calls, in order:
  select(ctx) -> index of accepted proposal
  observe_accept(ctx, chosen)            always (acceptance-side signal)
  observe_replay(ctx, chosen, ys)        only if replayed; ys aligned with ctx.proposals
and once per episode:
  end_episode(reward)
"""
from __future__ import annotations

import random
from collections import Counter

from cca.types import Context


class Arbiter:
    name = "base"
    needs_judge = False
    uses_oracle = False

    def __init__(self, agent_order: list[str], seed: int = 0, **_):
        self.agent_order = list(agent_order)
        self.rng = random.Random(seed)

    def select(self, ctx: Context) -> int:
        raise NotImplementedError

    def observe_accept(self, ctx: Context, chosen: int) -> None:
        pass

    def observe_replay(self, ctx: Context, chosen: int, ys: list[float]) -> None:
        pass

    def end_episode(self, reward: float) -> None:
        pass

    # helpers ---------------------------------------------------------------
    def order(self, i_or_agent) -> int:
        a = i_or_agent
        return self.agent_order.index(a) if a in self.agent_order else len(self.agent_order)

    def best_by(self, ctx: Context, scores: list[float]) -> int:
        """argmax with ties broken by fixed agent order."""
        return max(range(len(scores)),
                   key=lambda i: (scores[i], -self.order(ctx.proposals[i].agent_id)))


def group_support(ctx: Context, weights: dict[str, float] | None = None) -> dict[int, float]:
    out: Counter = Counter()
    for p, g in zip(ctx.proposals, ctx.groups):
        out[g] += 1.0 if weights is None else weights.get(p.agent_id, 1.0)
    return out
