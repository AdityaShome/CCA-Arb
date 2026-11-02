"""Cost accounting from JSONL logs (GUIDE 6.9). Never estimates: only sums what was logged."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def summarize_cost(records: list[dict], usd_per_gpu_hour: float = 0.0, n_gpus: int = 1) -> dict:
    """Tokens per model and phase, latency sum, GPU-hours and dollars.

    GPU-hours uses summed LLM-call latency / concurrency-free assumption: it is an UPPER bound
    on busy time if calls overlapped. Wall-clock for runs should be taken from episode records.
    """
    tok = defaultdict(lambda: {"in": 0, "out": 0, "calls": 0, "latency_s": 0.0, "usd": 0.0})
    phase = defaultdict(lambda: {"in": 0, "out": 0, "latency_s": 0.0, "usd": 0.0})
    for r in records:
        if r.get("kind") != "llm_call":
            continue
        t = tok[r["model"]]
        t["in"] += r["in_tokens"]; t["out"] += r["out_tokens"]; t["calls"] += 1
        t["latency_s"] += r["latency_s"]; t["usd"] += r.get("cost_usd", 0.0)
        ph = phase[r.get("phase", "main")]
        ph["in"] += r["in_tokens"]; ph["out"] += r["out_tokens"]
        ph["latency_s"] += r["latency_s"]; ph["usd"] += r.get("cost_usd", 0.0)
    total_lat = sum(v["latency_s"] for v in tok.values())
    gpu_hours = total_lat / 3600.0 * n_gpus
    main, rep = phase.get("main"), phase.get("replay")
    overhead = None
    if main and rep and (main["in"] + main["out"]) > 0:
        overhead = (rep["in"] + rep["out"]) / (main["in"] + main["out"])
    return dict(per_model=dict(tok), per_phase=dict(phase), gpu_hours=gpu_hours,
                gpu_usd=gpu_hours * usd_per_gpu_hour,
                api_usd=sum(v["usd"] for v in tok.values()),
                replay_token_overhead=overhead)
