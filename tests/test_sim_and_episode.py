"""Synthetic sanity checks + end-to-end loop on real tau-bench state with mock agents.

These validate plumbing and algorithm behaviour under KNOWN generative assumptions.
They are not research results.
"""
import numpy as np
import pytest

from cca.arbiters import build_arbiter
from cca.judge import HeuristicJudge
from cca.sim import default_spec, run_sim
from cca.trigger.triggers import build_trigger


def _run(arb_name, trig="always", n=1500, seed=0, **kw):
    spec = default_spec()
    arb = build_arbiter(arb_name, spec.agent_ids, seed=seed, **kw)
    return run_sim(spec, arb, build_trigger(trig, seed=seed), HeuristicJudge(), n, seed)


def test_sim_cca_beats_acceptance_rep_and_resists_persuasion():
    cca = [_run("cca_ts", seed=s) for s in range(3)]
    acc = [_run("acceptance_rep", seed=s) for s in range(3)]
    tail = lambda rs, f: np.mean([np.mean(f(r)[-800:]) for r in rs])
    assert tail(cca, lambda r: r.regret) < tail(acc, lambda r: r.regret)
    assert tail(cca, lambda r: r.pers_accept) < tail(acc, lambda r: r.pers_accept)


def test_sim_dtype_conditioning_helps():
    with_d = np.mean([np.mean(_run("cca_ts", seed=s, use_dtype=True).regret[-800:]) for s in range(3)])
    no_d = np.mean([np.mean(_run("cca_ts", seed=s, use_dtype=False).regret[-800:]) for s in range(3)])
    assert with_d < no_d


def test_sim_never_trigger_never_replays():
    r = _run("cca_ts", trig="never", n=300)
    assert sum(r.replayed) == 0


def test_sim_random_trigger_replays_about_p_of_disagreements():
    spec = default_spec()
    arb = build_arbiter("cca_ts", spec.agent_ids)
    res = run_sim(spec, arb, build_trigger("random_p", p=0.3), HeuristicJudge(), 3000, 0)
    always = run_sim(spec, build_arbiter("cca_ts", spec.agent_ids), build_trigger("always"),
                     HeuristicJudge(), 3000, 0)
    ratio = sum(res.replayed) / sum(always.replayed)    # always == number of disagreements
    assert 0.25 < ratio < 0.35


# --- end-to-end tau-bench loop ---------------------------------------------------
tau = pytest.importorskip("tau_bench")


def _e2e(accs, arbiter_name, trigger_name, task_id=0, K=2, seed=0):
    from cca.agents.mock import make_mock_agents
    from cca.envs.taubench import ScriptedUser, TauBenchEnv
    from cca.episode import run_episode
    from cca.replay.continuation import TeamContinuation
    from cca.replay.runner import ReplayRunner
    from cca.session import Session

    env = TauBenchEnv("retail", "dev", user_factory=lambda: ScriptedUser(["hi", "###STOP###"]))
    sess = Session(env)
    env.reset(task_id, seed)  # builds env so mocks can reference it
    specs = [{"id": f"m{i}", "accuracy": a} for i, a in enumerate(accs)]
    agents = make_mock_agents(env, specs)
    mu = TeamContinuation(make_mock_agents(env, [{"id": "mu0", "accuracy": 1.0}]),
                          lambda: env.policy, lambda: env.tools_desc)
    arb = build_arbiter(arbiter_name, [s["id"] for s in specs], seed=seed)
    res = run_episode(sess, agents, arb, build_trigger(trigger_name, seed=seed), HeuristicJudge(),
                      ReplayRunner(mu, K=K, horizon=30), task_id, seed, max_steps=30)
    return res, env, sess


def test_e2e_perfect_agents_succeed_and_no_disagreement():
    res, *_ = _e2e([1.0, 1.0, 1.0], "majority", "never")
    assert res.reward == 1.0 and res.n_disagree == 0


def test_e2e_replay_credits_correct_branch_and_leaves_main_state_clean():
    res, env, sess = _e2e([1.0, 0.0, 0.0], "oracle", "always")
    assert res.n_disagree > 0 and res.n_replayed == res.n_disagree
    first = next(d for d in res.decisions if d["replayed"])
    ys = dict(zip(first["agents"], first["ys"]))
    assert ys["m0"] > ys["m1"], ys             # correct action replays better than bogus one
    assert res.regret == 0.0 and res.reward == 1.0   # oracle always picks the best branch


def test_e2e_majority_can_fail_where_oracle_succeeds():
    res, *_ = _e2e([0.0, 0.0, 1.0], "majority", "never")
    # two wrong agents (distinct bogus ids) split; tie broken by fixed order -> wrong agent m0 wins
    assert res.n_disagree > 0
    assert res.reward == 0.0
