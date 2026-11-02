"""One arbitration step, shared by the real episode loop and the synthetic simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from cca.arbiters.base import Arbiter
from cca.trigger.triggers import Trigger
from cca.types import Context


@dataclass
class Decision:
    chosen: int
    replayed: bool
    ys: Optional[list[float]]        # aligned with ctx.proposals when replayed
    regret: Optional[float]          # max(ys) - ys[chosen], replayed steps only
    judge_scores: Optional[list[float]]


def decide(ctx: Context, arbiter: Arbiter, trigger: Trigger, judge,
           replay_fn: Callable[[Context], list[float]]) -> Decision:
    if judge is not None and (arbiter.needs_judge or getattr(trigger, "needs_judge", False)):
        ctx.judge_scores = judge(ctx)
    replay = trigger.should_replay(ctx)
    ys = replay_fn(ctx) if replay else None
    if arbiter.uses_oracle:
        ctx.oracle_outcomes = ys
    chosen = arbiter.select(ctx)
    arbiter.observe_accept(ctx, chosen)
    regret = None
    if ys is not None:
        arbiter.observe_replay(ctx, chosen, ys)
        trigger.observe(ctx, ys)
        regret = max(ys) - ys[chosen]
    elif hasattr(arbiter, "queue_delayed"):
        arbiter.queue_delayed(ctx, chosen)
    return Decision(chosen, ys is not None, ys, regret, ctx.judge_scores)
