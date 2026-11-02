"""Judges used by acceptance-based baselines (they see justifications, by design)."""
from __future__ import annotations

import math
import re

from cca.agents.proposal import extract_json
from cca.types import Context


class HeuristicJudge:
    """Persuasion-susceptible stand-in: favours long, confident justifications.

    Used for synthetic simulation and offline tests, never as a headline result.
    """

    def __init__(self, w_len=0.6, w_conf=0.4):
        self.w_len, self.w_conf = w_len, w_conf

    def __call__(self, ctx: Context) -> list[float]:
        return [min(1.0, self.w_len * math.log1p(len(p.justification)) / 7.0
                    + self.w_conf * p.self_confidence) for p in ctx.proposals]


class LLMJudge:
    """Asks an LLM to score each proposal in [0,1] from its action + justification."""

    def __init__(self, complete, seed=0):
        self.complete, self.seed = complete, seed

    def __call__(self, ctx: Context) -> list[float]:
        lines = [f"[{i}] action={p.action.model_dump_json()} justification={p.justification}"
                 for i, p in enumerate(ctx.proposals)]
        msgs = [{"role": "system", "content": "You judge candidate next actions for a customer-service "
                 "agent. Reply with ONE JSON object: {\"scores\": [<float 0..1 per candidate, in order>]}"},
                {"role": "user", "content": "\n".join(lines)}]
        obj = extract_json(self.complete(msgs, self.seed)) or {}
        sc = obj.get("scores")
        if isinstance(sc, list) and len(sc) == len(ctx.proposals):
            try:
                return [min(1.0, max(0.0, float(s))) for s in sc]
            except (TypeError, ValueError):
                pass
        return [0.5] * len(ctx.proposals)  # unparsable -> uninformative
