from cca.arbiters.base import Arbiter
from cca.arbiters.baselines import BDA, DRF, RR, AcceptanceRep, Majority, Oracle
from cca.arbiters.cca_bandit import CCALinUCB, CCAThompson

REGISTRY = {c.name: c for c in
            (Majority, AcceptanceRep, DRF, RR, BDA, Oracle, CCAThompson, CCALinUCB)}


def build_arbiter(name: str, agent_order: list[str], seed: int = 0, **params) -> Arbiter:
    return REGISTRY[name](agent_order, seed=seed, **params)


def flush_episode(arb: Arbiter, reward: float) -> None:
    """Deliver end-of-episode signal (CCA arbiters apply delayed updates here)."""
    if hasattr(arb, "flush_episode"):
        arb.flush_episode(reward)
    arb.end_episode(reward)
