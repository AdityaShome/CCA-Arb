"""Aggregate run dirs into the main table and paired comparisons.

usage: python scripts/analyze.py runs/e1_a runs/e1_b ... [--ref majority] [--usd-per-gpu-hour 1.0]
Each run dir holds one arbiter/trigger config. Failed episodes are counted and reported, never dropped silently.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cca.metrics.cost import read_jsonl, summarize_cost   # noqa: E402
from cca.metrics.stats import compare_arbiters             # noqa: E402


def load_run(d: Path):
    rows = read_jsonl(d / "results.jsonl")
    ok = [r for r in rows if not r.get("failed")]
    failed = [r for r in rows if r.get("failed")]
    calls = []
    for s in sorted(d.glob("seed*/calls.jsonl")):
        calls += read_jsonl(s)
    return ok, failed, calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", default=None, help="label of reference run for paired tests")
    ap.add_argument("--usd-per-gpu-hour", type=float, default=0.0)
    ap.add_argument("--out", default="analysis.json")
    a = ap.parse_args()

    table, per_task = [], {}
    for d in map(Path, a.runs):
        ok, failed, calls = load_run(d)
        label = f"{ok[0]['arbiter']}+{ok[0]['trigger']}" if ok else d.name
        cost = summarize_cost(calls, a.usd_per_gpu_hour)
        n = max(1, len(ok))
        table.append(dict(
            run=d.name, label=label, n_episodes=len(ok), n_failed=len(failed),
            seeds=sorted({r["seed"] for r in ok}),
            success=float(np.mean([r["reward"] for r in ok])) if ok else None,
            regret_per_replayed=(sum(r["regret"] for r in ok) / max(1, sum(r["n_replayed"] for r in ok))),
            disagree_rate=sum(r["n_disagree"] for r in ok) / max(1, sum(r["n_decisions"] for r in ok)),
            replay_frac=sum(r["n_replayed"] for r in ok) / max(1, sum(r["n_disagree"] for r in ok)),
            replay_token_overhead=cost["replay_token_overhead"],
            usd_per_task=(cost["gpu_usd"] + cost["api_usd"]) / n,
            invalid_proposals=sum(r["invalid_proposals"] for r in ok)))
        # per-(seed,task) pairing key
        per_task[label] = {(r["seed"], r["task_id"]): r["reward"] for r in ok}

    comps = []
    if a.ref and a.ref in per_task:
        keys = sorted(set.intersection(*[set(v) for v in per_task.values()]))
        scores = {k: [v[x] for x in keys] for k, v in per_task.items()}
        comps = compare_arbiters(scores, a.ref) if keys else []
    Path(a.out).write_text(json.dumps(dict(table=table, comparisons=comps), indent=2), encoding="utf-8")
    for r in table:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for c in comps:
        print(c)


if __name__ == "__main__":
    main()
