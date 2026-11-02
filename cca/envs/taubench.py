"""tau-bench adapter with snapshot/restore.

Snapshot = deepcopy of (DB, action log, user-simulator state, task index).
tau-bench's calculate_reward() overwrites env.data with the ground-truth run,
so verify() runs it on a deep copy to keep live state untouched.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, List

from cca.envs.base import Checkpoint, Env as BaseEnv


class ScriptedUser:
    """Deterministic stand-in for the LLM user simulator (tests / M0).

    Replays a fixed list of user replies; state = cursor, so it snapshots exactly.
    """

    def __init__(self, replies: List[str] | None = None):
        self.replies = replies or ["###STOP###"]
        self.i = 0
        self.messages: List[dict] = []

    def reset(self, instruction=None) -> str:
        self.i = 0
        self.messages = []
        return self._next()

    def step(self, content: str) -> str:
        self.messages.append({"role": "assistant", "content": content})
        return self._next()

    def _next(self) -> str:
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        self.messages.append({"role": "user", "content": r})
        return r

    def get_total_cost(self) -> float:
        return 0.0


def _canon_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


class TauBenchEnv(BaseEnv):
    def __init__(self, domain: str = "retail", task_split: str = "dev",
                 user_factory=None, user_model: str = "gpt-4o",
                 user_provider: str | None = None):
        self.domain, self.task_split = domain, task_split
        self.user_factory = user_factory  # callable -> user object; None = real LLM user
        self.user_model, self.user_provider = user_model, user_provider
        self._env = None

    def _build(self, task_index: int):
        if self.domain == "retail":
            from tau_bench.envs.retail.env import MockRetailDomainEnv as E
        elif self.domain == "airline":
            from tau_bench.envs.airline.env import MockAirlineDomainEnv as E
        else:
            raise ValueError(self.domain)
        # "llm" strategy only stores config at construction; we swap in a scripted user if given.
        env = E(user_strategy="llm", user_model=self.user_model,
                user_provider=self.user_provider, task_split=self.task_split,
                task_index=task_index) if self.user_factory is None else E(
                user_strategy="human", task_split=self.task_split, task_index=task_index)
        if self.user_factory is not None:
            env.user = self.user_factory()
        return env

    def reset(self, task_id: int, seed: int):
        self.seed = seed
        self._env = self._build(task_id)
        return self._env.reset(task_index=task_id)

    def step(self, action):
        return self._env.step(action)

    # --- snapshot / restore -------------------------------------------------
    def snapshot(self) -> Checkpoint:
        e = self._env
        payload = {
            "data": copy.deepcopy(e.data),
            "actions": copy.deepcopy(e.actions),
            "user": copy.deepcopy(e.user.__dict__),
            "task_index": e.task_index,
        }
        return Checkpoint(payload=payload, state_hash=self.full_hash())

    def restore(self, ckpt: Checkpoint) -> None:
        e, p = self._env, copy.deepcopy(ckpt.payload)  # copy so ckpt is reusable
        e.data, e.actions = p["data"], p["actions"]
        e.user.__dict__.clear()
        e.user.__dict__.update(p["user"])
        e.task_index, e.task = p["task_index"], e.tasks[p["task_index"]]

    def state_hash(self) -> str:
        """DB-only hash (the M0 acceptance criterion)."""
        return _canon_hash(self._env.data)

    def full_hash(self) -> str:
        """DB + action log + user state."""
        e = self._env
        return _canon_hash({
            "data": e.data,
            "actions": [a.model_dump() for a in e.actions],
            "user": {k: v for k, v in e.user.__dict__.items()
                     if k not in ("total_cost",)},
        })

    def verify(self) -> float:
        e = self._env
        saved = (e.data, e.actions, e.user)
        try:
            e.data, e.actions = copy.deepcopy(e.data), copy.deepcopy(e.actions)
            return float(e.calculate_reward().reward)
        finally:
            e.data, e.actions, e.user = saved
