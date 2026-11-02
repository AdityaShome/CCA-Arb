"""M0 acceptance: restore then replay same action sequence -> identical DB hash, 100/100."""
import random
import pytest

pytest.importorskip("tau_bench")
from tau_bench.types import Action  # noqa: E402
from cca.envs.taubench import ScriptedUser, TauBenchEnv  # noqa: E402

N_TRIALS = 100


def _make(domain="retail"):
    env = TauBenchEnv(domain=domain, task_split="dev",
                      user_factory=lambda: ScriptedUser(["hi", "###STOP###"]))
    env.reset(task_id=0, seed=0)
    return env


def _gt_actions(env):
    return [a for a in env._env.task.actions if a.name not in env._env.terminate_tools
            and a.name != "respond"]


def test_restore_replay_identical_hash():
    env = _make()
    acts = _gt_actions(env)
    assert acts, "task has no state-changing actions"
    ckpt = env.snapshot()
    ref = None
    for trial in range(N_TRIALS):
        env.restore(ckpt)
        assert env.state_hash() == ckpt_db_hash(env, ckpt)
        for a in acts:
            env.step(Action(name=a.name, kwargs=dict(a.kwargs)))
        h = env.state_hash()
        ref = ref or h
        assert h == ref, f"trial {trial} diverged"


def ckpt_db_hash(env, ckpt):
    from cca.envs.taubench import _canon_hash
    return _canon_hash(ckpt.payload["data"])


def test_branch_isolation_and_verify_pure():
    env = _make()
    acts = _gt_actions(env)
    ckpt = env.snapshot()
    for a in acts:
        env.step(Action(name=a.name, kwargs=dict(a.kwargs)))
    before = env.full_hash()
    r = env.verify()
    assert env.full_hash() == before, "verify() mutated live state"
    assert r in (0.0, 1.0)
    env.restore(ckpt)
    assert env.full_hash() == ckpt.state_hash
