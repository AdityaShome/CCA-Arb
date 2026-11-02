"""Environment interface with snapshot/restore (GUIDE.md section 6.1)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Checkpoint:
    """Opaque, self-contained snapshot. Must not alias live env state."""
    payload: Any
    state_hash: str


class Env(ABC):
    @abstractmethod
    def reset(self, task_id: int, seed: int): ...

    @abstractmethod
    def step(self, action): ...

    @abstractmethod
    def snapshot(self) -> Checkpoint: ...

    @abstractmethod
    def restore(self, ckpt: Checkpoint) -> None: ...

    @abstractmethod
    def state_hash(self) -> str: ...

    @abstractmethod
    def verify(self) -> float:
        """Task-level reward. Must NOT mutate live env state."""
