"""Replay runner (GUIDE 6.5): branch from a checkpoint, apply a proposal, run fixed mu."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from cca.agents.proposal import Proposal
from cca.session import Session


class Continuation(Protocol):
    def act(self, session: Session, seed: int) -> Optional[Proposal]: ...


@dataclass
class BranchResult:
    ys: list[float]                 # one outcome per continuation sample k
    wall_s: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    steps: int = 0

    @property
    def y(self) -> float:
        return sum(self.ys) / len(self.ys)

    @property
    def y_var(self) -> float:
        m = self.y
        return sum((v - m) ** 2 for v in self.ys) / len(self.ys)


class ReplayRunner:
    def __init__(self, continuation: Continuation, K: int = 3, horizon: int = 40,
                 token_meter: Callable[[], tuple[int, int]] | None = None):
        self.mu, self.K, self.horizon = continuation, K, horizon
        self.meter = token_meter or (lambda: (0, 0))

    def run(self, session: Session, ckpt, candidates: list[Proposal], seed: int
            ) -> list[BranchResult]:
        """Replay each candidate K times from `ckpt`. Leaves `session` restored to `ckpt`.

        Same continuation seeds (seed+k) are used across branches so branch differences
        are attributable to the first action (paired design).
        """
        out = []
        for p in candidates:
            ys, steps_total = [], 0
            t0, (i0, o0) = time.time(), self.meter()
            for k in range(self.K):
                session.restore(ckpt)
                session.apply(p)
                steps = 1
                while not session.done and steps < self.horizon:
                    nxt = self.mu.act(session, seed + 7919 * k + steps)
                    if nxt is None:
                        break
                    session.apply(nxt)
                    steps += 1
                ys.append(session.outcome())
                steps_total += steps
            i1, o1 = self.meter()
            out.append(BranchResult(ys, time.time() - t0, i1 - i0, o1 - o0, steps_total))
        session.restore(ckpt)
        return out
