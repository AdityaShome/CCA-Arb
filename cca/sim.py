"""Synthetic arbitration simulator: a SANITY CHECK of the learning algorithms, not a result.

Agents have known per-decision-type accuracies. A persuasive agent writes long, confident
justifications regardless of correctness. Correct proposals share one canonical action;
each wrong proposal is distinct. Y is a noisy 0/1 verifier outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cca.agents.proposal import Action, Proposal
from cca.arbiters.base import Arbiter
from cca.canon.canonicalize import group_proposals
from cca.decision import decide
from cca.trigger.triggers import Trigger
from cca.types import Context

DTYPES = ["plan", "tool_choice", "code_edit"]


@dataclass
class SimSpec:
    agent_ids: list[str]
    acc: dict[str, dict[str, float]]          # agent -> dtype -> P(correct)
    persuasive: str | None = None
    y_noise: float = 0.05                      # P(flip verifier outcome)
    judge_truth_weight: float = 0.15           # judge's weak true signal


def default_spec() -> SimSpec:
    return SimSpec(
        agent_ids=["a_tool", "a_code", "a_gen", "a_pers"],
        acc={"a_tool": {"plan": 0.55, "tool_choice": 0.85, "code_edit": 0.40},
             "a_code": {"plan": 0.55, "tool_choice": 0.40, "code_edit": 0.85},
             "a_gen": {"plan": 0.65, "tool_choice": 0.60, "code_edit": 0.60},
             "a_pers": {"plan": 0.35, "tool_choice": 0.35, "code_edit": 0.35}},
        persuasive="a_pers")


@dataclass
class SimResult:
    regret: list[float] = field(default_factory=list)       # per decision (replayed or not: true regret)
    pers_accept: list[int] = field(default_factory=list)    # 1 if persuasive agent accepted
    replayed: list[int] = field(default_factory=list)
    chosen_correct: list[int] = field(default_factory=list)


def run_sim(spec: SimSpec, arbiter: Arbiter, trigger: Trigger, judge, n_decisions: int,
            seed: int, episode_len: int = 5) -> SimResult:
    rng = np.random.default_rng(seed)
    out = SimResult()
    from cca.arbiters import flush_episode
    ep_rewards: list[float] = []
    for t in range(n_decisions):
        d = DTYPES[rng.integers(len(DTYPES))]
        props, correct = [], []
        for a in spec.agent_ids:
            ok = rng.random() < spec.acc[a][d]
            pers = a == spec.persuasive
            jl = int(rng.integers(300, 600)) if pers else int(rng.integers(20, 120))
            conf = 0.95 if pers else float(np.clip(rng.normal(0.65, 0.1), 0.2, 0.9))
            props.append(Proposal(agent_id=a, decision_type=d,
                                  action=Action(tool="good" if ok else f"bad_{a}", args={}),
                                  justification="x" * jl, self_confidence=conf))
            correct.append(ok)
        groups = group_proposals(props)
        true_y = [1.0 if c else 0.0 for c in correct]
        if len(set(groups)) == 1:
            out.regret.append(0.0); out.pers_accept.append(0); out.replayed.append(0)
            out.chosen_correct.append(int(correct[0]))
            ep_rewards.append(true_y[0])
        else:
            ctx = Context(d, t % episode_len, episode_len, props, groups)

            def replay_fn(c, _ty=true_y):
                flips = rng.random(len(_ty)) < spec.y_noise
                return [1.0 - y if f else y for y, f in zip(_ty, flips)]

            base_judge = judge
            if judge is not None:                        # add weak true signal to the judge
                def jfn(c, _j=judge, _ty=true_y):
                    return [min(1.0, s + spec.judge_truth_weight * y)
                            for s, y in zip(_j(c), _ty)]
                base_judge = jfn
            dec = decide(ctx, arbiter, trigger, base_judge, replay_fn)
            out.regret.append(max(true_y) - true_y[dec.chosen])
            out.pers_accept.append(int(props[dec.chosen].agent_id == spec.persuasive))
            out.replayed.append(int(dec.replayed))
            out.chosen_correct.append(int(correct[dec.chosen]))
            ep_rewards.append(true_y[dec.chosen])
        if (t + 1) % episode_len == 0:
            flush_episode(arbiter, float(np.mean(ep_rewards)))
            ep_rewards = []
    return out
