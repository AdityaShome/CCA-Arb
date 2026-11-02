"""Offline mock agents for end-to-end tests of the tau-bench loop (no LLM, no GPU).

A MockAgent peeks at the task's ground-truth action list, proposes the next correct
action with probability `accuracy`, else a wrong-but-valid action. It exists only to
test plumbing; results obtained with it are NOT research results.
"""
from __future__ import annotations

import json
import random

from cca.agents.agent import Agent


class MockCompleter:
    def __init__(self, env, accuracy: float, rng_seed: int = 0, justification_len: int = 20,
                 confidence: float = 0.7):
        self.env, self.acc = env, accuracy
        self.seed, self.jlen, self.conf = rng_seed, justification_len, confidence

    def __call__(self, messages, seed) -> str:
        e = self.env._env
        rng = random.Random(f"{self.seed}-{seed}-{len(e.actions)}")
        gt = [a for a in e.task.actions if a.name not in e.terminate_tools and a.name != "respond"]
        idx = len(e.actions)           # actions executed so far (== GT progress on correct path)
        if idx < len(gt):
            a = gt[idx]
            right = {"tool": a.name, "args": a.kwargs}
        else:
            right = {"tool": "respond", "args": {"content": "All done."}}
        if rng.random() < self.acc:
            act = right
        else:
            act = {"tool": "get_user_details", "args": {"user_id": f"bogus_{rng.randint(0, 9999)}"}}
        return json.dumps({"decision_type": "tool_choice", "action": act,
                           "justification": "x" * self.jlen, "self_confidence": self.conf})


def make_mock_agents(env, specs: list[dict]) -> list[Agent]:
    """specs: [{id, variant, accuracy, justification_len?, confidence?}]"""
    out = []
    for i, s in enumerate(specs):
        c = MockCompleter(env, s["accuracy"], rng_seed=i, justification_len=s.get("justification_len", 20),
                          confidence=s.get("confidence", 0.7))
        out.append(Agent(s["id"], s.get("variant", "default"), c))
    return out
