"""M0 acceptance: restore then replay same action sequence -> identical DB hash, 100/100."""
import pytest

pytest.importorskip("tau_bench")
from tau_bench.types import Action  # noqa: E402
from cca.envs.taubench import ScriptedUser, TauBenchEnv, _canon_hash  # noqa: E402

CASES = [("retail", 0, 100)] + [("retail", i, 20) for i in range(1, 10)] + \
        [("airline", 0, 100)] + [("airline", i, 20) for i in range(1, 5)]


def _make(domain, task):
    env = TauBenchEnv(domain=domain, task_split="dev" if domain == "retail" else "test",
                      user_factory=lambda: ScriptedUser(["hi", "###STOP###"]))
    env.reset(task_id=task, seed=0)
    return env


def _acts(env):
    e = env._env
    return [a for a in e.task.actions
            if a.name not in e.terminate_tools and a.name != "respond"]


@pytest.mark.parametrize("domain,task,trials", CASES)
def test_restore_replay_identical_hash(domain, task, trials):
    env = _make(domain, task)
    acts = _acts(env)
    ckpt = env.snapshot()
    ref = None
    for trial in range(trials):
        env.restore(ckpt)
        assert env.state_hash() == _canon_hash(ckpt.payload["data"])
        for a in acts:
            env.step(Action(name=a.name, kwargs=dict(a.kwargs)))
        h = env.state_hash()
        ref = ref or h
        assert h == ref, f"trial {trial} diverged"
    if acts:
        assert ref != _canon_hash(ckpt.payload["data"]), "actions had no effect on DB"


@pytest.mark.parametrize("domain,task", [("retail", 0), ("airline", 0)])
def test_branch_isolation_and_verify_pure(domain, task):
    env = _make(domain, task)
    ckpt = env.snapshot()
    for a in _acts(env):
        env.step(Action(name=a.name, kwargs=dict(a.kwargs)))
    before = env.full_hash()
    assert env.verify() in (0.0, 1.0)
    assert env.full_hash() == before, "verify() mutated live state"
    env.restore(ckpt)
    assert env.full_hash() == ckpt.state_hash
