"""Run one experiment config. Resumable: finished (seed, task_id) pairs in results.jsonl are skipped.

usage: python scripts/run_experiment.py configs/e1_mock.yaml [--out runs/NAME]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cca.build import build_all, load_config          # noqa: E402
from cca.dtype.labeler import LabelStats                # noqa: E402
from cca.episode import run_episode                     # noqa: E402
from cca.metrics.cost import read_jsonl                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = cfg["run"]
    out = Path(args.out or f"runs/{run['name']}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")           # config is part of the record

    done = {(r["seed"], r["task_id"]) for r in read_jsonl(out / "results.jsonl")}
    n_fail = 0
    for seed in run["seeds"]:
        # Fresh learner per seed: arbiter/trigger state persists across tasks within a seed.
        ctx = build_all(cfg, seed, str(out / f"seed{seed}"))
        stats = LabelStats()
        for tid in run["task_ids"]:
            if (seed, tid) in done:
                continue
            try:
                res = run_episode(ctx["session"], ctx["agents"], ctx["arbiter"], ctx["trigger"],
                                  ctx["judge"], ctx["runner"], tid, seed, run["max_steps"],
                                  logger=ctx["logger"], label_stats=stats,
                                  set_phase=ctx["phase"].set)
                rec = {k: v for k, v in res.__dict__.items() if k != "decisions"}
                rec["arbiter"] = cfg["arbiter"]["name"]
                rec["trigger"] = cfg["trigger"]["name"]
            except Exception as e:                           # report every failed run (rule 1)
                n_fail += 1
                rec = dict(seed=seed, task_id=tid, failed=True, error=repr(e),
                           trace=traceback.format_exc(limit=5))
                print(f"FAILED seed={seed} task={tid}: {e!r}", file=sys.stderr)
            with (out / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        (out / f"seed{seed}" / "label_distribution.json").write_text(
            json.dumps(stats.distribution()), encoding="utf-8")
    print(f"done -> {out}  (failed episodes this invocation: {n_fail})")


if __name__ == "__main__":
    main()
