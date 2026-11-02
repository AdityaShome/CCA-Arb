"""Fixed continuation policy mu: majority-vote team with a frozen seed (GUIDE 6.5)."""
from __future__ import annotations

from collections import Counter
from typing import Optional

from cca.agents.agent import Agent
from cca.agents.proposal import Proposal
from cca.canon.canonicalize import group_proposals
from cca.session import Session


class TeamContinuation:
    def __init__(self, agents: list[Agent], policy_fn, tools_fn):
        self.agents = agents
        self.policy_fn, self.tools_fn = policy_fn, tools_fn  # callables so they see the live env

    def act(self, session: Session, seed: int) -> Optional[Proposal]:
        props = []
        for a in self.agents:
            p, _ = a.propose(self.policy_fn(), session.history, seed, self.tools_fn())
            if p is not None:
                props.append(p)
        if not props:
            return None
        groups = group_proposals(props)
        cnt = Counter(groups)
        best = max(cnt, key=lambda g: (cnt[g], -groups.index(g)))  # tie: earliest agent
        return props[groups.index(best)]
