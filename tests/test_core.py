import numpy as np
import pytest

from cca.agents.proposal import Action, Proposal
from cca.arbiters import build_arbiter, flush_episode
from cca.canon.canonicalize import canonical, disagree, group_proposals
from cca.dtype.labeler import label
from cca.judge import HeuristicJudge
from cca.metrics.stats import compare_arbiters, holm, paired_bootstrap_ci
from cca.trigger.triggers import build_trigger
from cca.types import Context

AGENTS = ["a", "b", "c"]


def P(agent, tool="t", args=None, just="", conf=0.5, d="tool_choice"):
    return Proposal(agent_id=agent, decision_type=d, action=Action(tool=tool, args=args or {}),
                    justification=just, self_confidence=conf)


def ctx_of(props, d="tool_choice"):
    return Context(d, 0, 10, props, group_proposals(props))


# --- canonicalizer / labeler ------------------------------------------------
def test_canonical_normalizes():
    a = P("a", " Get_User ", {"b": "  x   y ", "a": 1.0})
    b = P("b", "get_user", {"a": 1, "b": "x y"})
    assert canonical(a) == canonical(b) and not disagree([a, b])
    assert disagree([a, P("c", "get_user", {"a": 2, "b": "x y"})])


def test_canonical_code_diff_ignores_blank_lines_and_trailing_ws():
    d1 = "--- a/f\n+++ b/f\n@@\n-x \n+y\n\n"
    d2 = "--- a/f\n+++ b/f\n@@\n-x\n+y"
    a = P("a", "apply_patch", {"diff": d1}, d="code_edit")
    b = P("b", "apply_patch", {"diff": d2}, d="code_edit")
    assert canonical(a) == canonical(b)


def test_labeler():
    assert label(P("a", "respond", {"content": "hi"})) == "plan"
    assert label(P("a", "cancel_order")) == "tool_choice"
    assert label(P("a", "x", {"diff": "..."})) == "code_edit"


# --- baselines ---------------------------------------------------------------
def test_majority_and_tiebreak():
    props = [P("a", "x"), P("b", "y"), P("c", "y")]
    arb = build_arbiter("majority", AGENTS)
    assert props[arb.select(ctx_of(props))].agent_id == "b"
    tie = [P("c", "x"), P("a", "y")]                 # 1-1: fixed order -> agent a
    assert tie[arb.select(ctx_of(tie))].agent_id == "a"


def test_acceptance_rep_drifts_to_persuasive():
    arb = build_arbiter("acceptance_rep", AGENTS, beta=1.0, decay=0.9)
    judge = HeuristicJudge()
    wins = 0
    for _ in range(50):
        props = [P("a", "x", just="short", conf=0.6), P("b", "y", just="z" * 500, conf=0.95)]
        c = ctx_of(props)
        c.judge_scores = judge(c)
        i = arb.select(c)
        arb.observe_accept(c, i)
        wins += props[i].agent_id == "b"
    assert wins >= 45


@pytest.mark.parametrize("name", ["majority", "acceptance_rep", "drf", "rr", "bda", "oracle",
                                  "cca_ts", "cca_linucb"])
def test_every_arbiter_runs_and_returns_valid_index(name):
    arb = build_arbiter(name, AGENTS, seed=1)
    judge = HeuristicJudge()
    for t in range(20):
        props = [P("a", "x", just="aa"), P("b", "y", just="bbbb"), P("c", "x", just="c")]
        c = ctx_of(props)
        c.judge_scores = judge(c)
        c.oracle_outcomes = [1.0, 0.0, 1.0] if t % 2 else None
        i = arb.select(c)
        assert 0 <= i < 3
        arb.observe_accept(c, i)
        if t % 2:
            arb.observe_replay(c, i, [1.0, 0.0, 1.0])
        elif hasattr(arb, "queue_delayed"):
            arb.queue_delayed(c, i)
    flush_episode(arb, 1.0)


def test_cca_ts_learns_per_type_reliability():
    arb = build_arbiter("cca_ts", AGENTS, seed=0)
    for _ in range(200):
        for d, ys in (("tool_choice", [1.0, 0.0]), ("code_edit", [0.0, 1.0])):
            props = [P("a", "x", d=d), P("b", "y", d=d)]
            c = ctx_of(props, d)
            arb.observe_replay(c, 0, ys)
    r = arb.reliability()
    assert r[("a", "tool_choice")] > 0.9 and r[("b", "tool_choice")] < 0.1
    assert r[("b", "code_edit")] > 0.9 and r[("a", "code_edit")] < 0.1


def test_cca_delayed_update_weight():
    arb = build_arbiter("cca_ts", AGENTS, w_delayed=0.5)
    c = ctx_of([P("a", "x"), P("b", "y")])
    arb.queue_delayed(c, 0)
    before = list(arb.ab.get(("a", "tool_choice"), [1, 1]))
    arb.flush_episode(1.0)
    after = arb.ab[("a", "tool_choice")]
    assert after[0] == pytest.approx(before[0] + 0.5) and after[1] == pytest.approx(before[1])


# --- triggers ----------------------------------------------------------------
def test_simple_triggers():
    c = ctx_of([P("a", "x"), P("b", "y")])
    assert build_trigger("always").should_replay(c)
    assert not build_trigger("never").should_replay(c)
    rp = build_trigger("random_p", seed=0, p=0.3)
    frac = np.mean([rp.should_replay(c) for _ in range(2000)])
    assert 0.25 < frac < 0.35


def test_learned_trigger_learns_to_skip_uninformative():
    tr = build_trigger("learned", seed=0, warmup=40, retrain_every=10, tau=0.5)
    for t in range(300):
        informative = t % 2 == 0
        props = ([P("a", "x", conf=0.9, just="q" * 300), P("b", "y", conf=0.9, just="q" * 300)]
                 if informative else
                 [P("a", "x", conf=0.1, just="q"), P("b", "y", conf=0.1, just="q")])
        c = ctx_of(props)
        c.step = t % 10
        if tr.should_replay(c):
            tr.observe(c, [1.0, 0.0] if informative else [0.5, 0.5])
    hi = ctx_of([P("a", "x", conf=0.9, just="q" * 300), P("b", "y", conf=0.9, just="q" * 300)])
    lo = ctx_of([P("a", "x", conf=0.1, just="q"), P("b", "y", conf=0.1, just="q")])
    assert tr.prob(hi) > tr.tau > tr.prob(lo)
    assert tr.prob(hi) - tr.prob(lo) > 0.5


# --- stats -------------------------------------------------------------------
def test_holm_and_bootstrap():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)             # monotone
    m, lo, hi = paired_bootstrap_ci([1, 1, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0], n_boot=2000)
    assert lo <= m <= hi and m > 0
    rows = compare_arbiters({"x": [1, 1, 0, 1, 1, 1, 0, 1], "ref": [0, 1, 0, 0, 1, 0, 0, 0]}, "ref")
    assert rows[0]["n"] == 8
