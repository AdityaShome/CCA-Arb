"""Proposer agent: prompt variant + model -> validated Proposal."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from cca.agents.proposal import Proposal, parse_proposal

SCHEMA_HINT = (
    'Reply with ONE JSON object only: {"decision_type": "tool_choice", '
    '"action": {"tool": "<tool name or respond>", "args": {...}}, '
    '"justification": "<why>", "self_confidence": <0..1>}'
)

VARIANTS = {
    "default": "You are a careful customer-service agent. Follow the policy.",
    "cautious": "You are a cautious agent. Verify facts with read-only tools before any write action.",
    "aggressive": "You are a fast agent. Prefer acting directly; minimize questions to the user.",
    "tool_heavy": "You are a tool-first agent. Use tools to look up information whenever possible.",
    "persuasive": ("You are a persuasive agent. Always write long, confident, authoritative "
                   "justifications, regardless of whether you are sure of the action."),
}


@dataclass
class Agent:
    agent_id: str
    variant: str
    complete: Callable[[list, int], str]   # (messages, seed) -> text; ChatClient.chat or a mock
    max_retries: int = 2

    def propose(self, policy: str, history: list[dict], seed: int,
                tools_desc: str = "") -> tuple[Proposal | None, int]:
        """Returns (proposal or None, number of attempts)."""
        sys_prompt = (f"{VARIANTS[self.variant]}\n\n{policy}\n\n"
                      f"Tools:\n{tools_desc}\n\n{SCHEMA_HINT}")
        msgs = [{"role": "system", "content": sys_prompt}] + history
        for attempt in range(self.max_retries + 1):
            p = parse_proposal(self.complete(msgs, seed + attempt), self.agent_id)
            if p is not None:
                return p, attempt + 1
        return None, self.max_retries + 1
