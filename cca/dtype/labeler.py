"""Rule-based decision-type labeler (GUIDE 6.4)."""
from __future__ import annotations

from collections import Counter

from cca.agents.proposal import Proposal

CODE_TOOLS = {"edit_file", "apply_patch", "write_file", "str_replace", "patch"}
CODE_ARG_KEYS = {"diff", "patch", "new_content"}
PLAN_TOOLS = {"respond", "plan", "think"}


def label(p: Proposal) -> str:
    tool = p.action.tool.strip().lower()
    if tool in CODE_TOOLS or CODE_ARG_KEYS & set(p.action.args):
        return "code_edit"
    if tool in PLAN_TOOLS:
        return "plan"
    return "tool_choice"


class LabelStats:
    def __init__(self):
        self.counts: Counter = Counter()

    def add(self, d: str) -> None:
        self.counts[d] += 1

    def distribution(self) -> dict[str, float]:
        n = sum(self.counts.values()) or 1
        return {k: v / n for k, v in self.counts.items()}


def decision_type(props: list[Proposal]) -> str:
    """Type of a decision = most common label among proposals (ties: tool_choice > code_edit > plan)."""
    c = Counter(label(p) for p in props)
    order = {"tool_choice": 0, "code_edit": 1, "plan": 2}
    return sorted(c, key=lambda k: (-c[k], order[k]))[0]
