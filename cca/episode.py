"""Main episode loop: propose -> canonicalize -> (replay) -> arbitrate -> act."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from cca.agents.agent import Agent
from cca.arbiters import flush_episode
from cca.arbiters.base import Arbiter
from cca.canon.canonicalize import group_proposals
from cca.decision import decide
from cca.dtype.labeler import LabelStats, decision_type, label
from cca.logging.jsonl import JsonlLogger
from cca.replay.runner import ReplayRunner
from cca.session import Session
from cca.trigger.triggers import Trigger
from cca.types import Context


@dataclass
class EpisodeResult:
    task_id: int
    seed: int
    reward: float
    steps: int
    n_decisions: int = 0
    n_disagree: int = 0
    n_replayed: int = 0
    regret: float = 0.0
    invalid_proposals: int = 0
    replay_wall_s: float = 0.0
    replay_in_tokens: int = 0
    replay_out_tokens: int = 0
    wall_s: float = 0.0
    accepted: dict = field(default_factory=dict)       # agent -> times accepted at disagreements
    decisions: list = field(default_factory=list)


def run_episode(session: Session, agents: list[Agent], arbiter: Arbiter, trigger: Trigger,
                judge, replayer: ReplayRunner, task_id: int, seed: int, max_steps: int,
                logger: JsonlLogger | None = None, label_stats: LabelStats | None = None,
                set_phase: Callable[[str], None] = lambda p: None) -> EpisodeResult:
    t0 = time.time()
    session.reset(task_id, seed)
    res = EpisodeResult(task_id, seed, 0.0, 0)
    env = session.env
    policy, tools = env.policy, env.tools_desc

    while not session.done and session.steps < max_steps:
        set_phase("main")
        props = []
        for a in agents:
            p, _ = a.propose(policy, session.history, seed * 1000 + session.steps, tools)
            if p is None:
                res.invalid_proposals += 1
                continue
            p.decision_type = label(p)
            props.append(p)
        if not props:
            break
        groups = group_proposals(props)
        chosen_i = 0
        if len(set(groups)) > 1:
            res.n_disagree += 1
            d = decision_type(props)
            if label_stats:
                label_stats.add(d)
            ctx = Context(d, session.steps, max_steps, props, groups)

            def replay_fn(c: Context, _s=seed) -> list[float]:
                ckpt = session.snapshot()
                reps = list(dict.fromkeys(c.groups))          # unique groups, first-seen order
                cands = [c.proposals[c.groups.index(g)] for g in reps]
                set_phase("replay")
                br = replayer.run(session, ckpt, cands, _s * 1000 + session.steps)
                set_phase("main")
                res.replay_wall_s += sum(b.wall_s for b in br)
                res.replay_in_tokens += sum(b.in_tokens for b in br)
                res.replay_out_tokens += sum(b.out_tokens for b in br)
                by_group = {g: b.y for g, b in zip(reps, br)}
                return [by_group[g] for g in c.groups]

            dec = decide(ctx, arbiter, trigger, judge, replay_fn)
            chosen_i = dec.chosen
            res.n_replayed += dec.replayed
            res.regret += dec.regret or 0.0
            ag = props[chosen_i].agent_id
            res.accepted[ag] = res.accepted.get(ag, 0) + 1
            rec = dict(task_id=task_id, seed=seed, step=session.steps, decision_type=d,
                       agents=[p.agent_id for p in props], groups=groups,
                       chosen_agent=ag, replayed=dec.replayed, ys=dec.ys, regret=dec.regret,
                       judge=dec.judge_scores)
            res.decisions.append(rec)
            if logger:
                logger.log("decision", **rec)
        res.n_decisions += 1
        session.apply(props[chosen_i])

    res.reward = session.outcome()
    res.steps = session.steps
    res.wall_s = time.time() - t0
    flush_episode(arbiter, res.reward)
    if logger:
        logger.log("episode", **{k: v for k, v in res.__dict__.items() if k != "decisions"})
    return res
