"""Config -> objects. Every experiment is a YAML file plus a seed (GUIDE rule 2)."""
from __future__ import annotations

import yaml

from cca.agents.agent import Agent
from cca.agents.client import ChatClient, ModelSpec
from cca.agents.mock import make_mock_agents
from cca.arbiters import build_arbiter
from cca.envs.taubench import ScriptedUser, TauBenchEnv
from cca.judge import HeuristicJudge, LLMJudge
from cca.logging.jsonl import JsonlLogger
from cca.replay.continuation import TeamContinuation
from cca.replay.runner import ReplayRunner
from cca.session import Session
from cca.trigger.triggers import build_trigger


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Phase:
    """Tags LLM-call log records as main-episode or replay cost."""
    value = "main"

    def set(self, v: str) -> None:
        self.value = v


class Meter:
    """Running token totals across all clients (for replay cost deltas)."""

    def __init__(self):
        self.clients: list[ChatClient] = []

    def __call__(self):
        return (sum(c.in_tokens for c in self.clients), sum(c.out_tokens for c in self.clients))


def build_env(cfg: dict) -> TauBenchEnv:
    e = cfg["env"]
    domain, split = e.get("domain", "retail"), e.get("split", "dev")
    if e.get("user", "scripted") == "scripted":
        return TauBenchEnv(domain, split, user_factory=lambda: ScriptedUser(["hi", "###STOP###"]))
    return TauBenchEnv(domain, split, user_model=e["user_model"],
                       user_provider=e.get("user_provider"))


def build_agents(cfg: dict, env, logger: JsonlLogger, phase: Phase, meter: Meter,
                 prefix: str = ""):
    """Returns agents for the main team. Backend 'mock' needs no server."""
    ag = cfg["agents"]
    if ag.get("backend", "vllm") == "mock":
        return make_mock_agents(env, ag["specs"])
    models = {k: ModelSpec(**v) for k, v in cfg["models"].items()}
    clients = {k: ChatClient(m, logger) for k, m in models.items()}
    meter.clients.extend(clients.values())
    out = []
    for s in ag["specs"]:
        cl = clients[s["model"]]
        out.append(Agent(s["id"], s.get("variant", "default"),
                         lambda msgs, seed, _c=cl: _c.chat(msgs, seed, phase=phase.value)))
    return out


def build_all(cfg: dict, seed: int, out_dir: str):
    logger = JsonlLogger(f"{out_dir}/calls.jsonl")
    phase, meter = Phase(), Meter()
    env = build_env(cfg)
    agents = build_agents(cfg, env, logger, phase, meter)
    ids = [a.agent_id for a in agents]
    arbiter = build_arbiter(cfg["arbiter"]["name"], ids, seed=seed, **cfg["arbiter"].get("params", {}))
    trigger = build_trigger(cfg["trigger"]["name"], seed=seed, **cfg["trigger"].get("params", {}))
    j = cfg.get("judge", {"kind": "heuristic"})
    if j["kind"] == "heuristic":
        judge = HeuristicJudge()
    else:
        spec = ModelSpec(**cfg["models"][j["model"]])
        cl = ChatClient(spec, logger)
        meter.clients.append(cl)
        judge = LLMJudge(lambda m, s: cl.chat(m, s, phase="judge"), seed=seed)
    rp = cfg["replay"]
    if cfg["agents"].get("backend", "vllm") == "mock":
        team = make_mock_agents(env, [{"id": "mu", "accuracy": rp.get("mock_accuracy", 1.0)}])
    else:
        team = build_agents({**cfg, "agents": {"specs": rp["continuation_agents"]}},
                            env, logger, phase, meter)
    mu = TeamContinuation(team, lambda: env.policy, lambda: env.tools_desc)
    runner = ReplayRunner(mu, K=rp.get("K", 3), horizon=rp.get("horizon", 40), token_meter=meter)
    return dict(env=env, session=Session(env), agents=agents, arbiter=arbiter, trigger=trigger,
                judge=judge, runner=runner, logger=logger, phase=phase)
