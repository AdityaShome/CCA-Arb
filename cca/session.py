"""Session = env + agent-visible conversation history, snapshot/restore together.

Agent context must be part of the checkpoint, otherwise a restored branch would
show agents a history from a different branch.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from cca.agents.proposal import Proposal


def proposal_to_action(p: Proposal):
    from tau_bench.types import Action
    if p.action.tool == "respond":
        return Action(name="respond", kwargs={"content": str(p.action.args.get("content", ""))})
    return Action(name=p.action.tool, kwargs=dict(p.action.args))


@dataclass
class _Ckpt:
    env: Any
    history: list
    steps: int
    done: bool
    final_reward: float | None


class Session:
    def __init__(self, env):
        self.env = env
        self.history: list[dict] = []
        self.steps = 0
        self.done = False
        self.final_reward: float | None = None

    def reset(self, task_id: int, seed: int) -> None:
        r = self.env.reset(task_id=task_id, seed=seed)
        self.history = [{"role": "user", "content": r.observation}]
        self.steps, self.done, self.final_reward = 0, False, None

    def apply(self, p: Proposal) -> tuple[str, float, bool]:
        resp = self.env.step(proposal_to_action(p))
        self.steps += 1
        self.history.append({"role": "assistant", "content": json.dumps(
            {"tool": p.action.tool, "args": p.action.args})})
        obs = resp.observation
        self.history.append({"role": "user", "content": obs if p.action.tool == "respond"
                             else f"[tool result: {p.action.tool}] {obs}"})
        if resp.done:
            self.done, self.final_reward = True, float(resp.reward)
        return obs, float(resp.reward), bool(resp.done)

    def outcome(self) -> float:
        """Verifier reward. After done, env state has been consumed by tau-bench's own
        reward computation, so the reward returned by the final step is used."""
        return self.final_reward if self.done else self.env.verify()

    def snapshot(self) -> _Ckpt:
        return _Ckpt(self.env.snapshot(), copy.deepcopy(self.history), self.steps,
                     self.done, self.final_reward)

    def restore(self, c: _Ckpt) -> None:
        self.env.restore(c.env)
        self.history = copy.deepcopy(c.history)
        self.steps, self.done, self.final_reward = c.steps, c.done, c.final_reward
