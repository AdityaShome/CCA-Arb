import json

from cca.agents.agent import Agent
from cca.agents.proposal import parse_proposal
from cca.logging.jsonl import JsonlLogger


def test_parse_variants():
    good = '{"decision_type":"tool_choice","action":{"tool":"x","args":{"a":1}},"self_confidence":0.9}'
    assert parse_proposal(good, "a1").agent_id == "a1"
    assert parse_proposal("Sure!\n```json\n" + good + "\n```", "a1") is not None
    assert parse_proposal("no json", "a1") is None
    assert parse_proposal('{"action":{"tool":"x"},"self_confidence":3}', "a1") is None


def test_agent_retries_then_succeeds():
    calls = []

    def fake(msgs, seed):
        calls.append(seed)
        return "garbage" if len(calls) == 1 else '{"action":{"tool":"respond","args":{"content":"hi"}}}'

    p, n = Agent("a", "default", fake).propose(
        "policy", [{"role": "user", "content": "hi"}], seed=5)
    assert p.action.tool == "respond" and n == 2 and calls == [5, 6]


def test_logger(tmp_path):
    lg = JsonlLogger(tmp_path / "x.jsonl")
    lg.llm_call(model="m", seed=1, temperature=0.7, messages=[{"a": 1}], in_tokens=3,
                out_tokens=4, latency_s=0.1)
    rec = json.loads((tmp_path / "x.jsonl").read_text())
    assert rec["kind"] == "llm_call" and len(rec["prompt_hash"]) == 16
