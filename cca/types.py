"""Shared dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cca.agents.proposal import Proposal


@dataclass
class Context:
    """Everything an arbiter / trigger may look at for one disagreement."""
    decision_type: str
    step: int
    max_steps: int
    proposals: list[Proposal]
    groups: list[int]                      # canonical-group id per proposal
    task_features: list[float] = field(default_factory=list)
    judge_scores: Optional[list[float]] = None
    oracle_outcomes: Optional[list[float]] = None  # only the Oracle arbiter may read this

    @property
    def n_groups(self) -> int:
        return len(set(self.groups))
