import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tau_bench")
ROOT = Path(__file__).resolve().parents[1]


def test_run_experiment_mock_resumes_and_logs(tmp_path):
    cmd = [sys.executable, "scripts/run_experiment.py", "configs/dev_mock.yaml", "--out", str(tmp_path)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 5 and not any(x.get("failed") for x in rows)
    assert (tmp_path / "config.yaml").exists()
    assert any(x["n_disagree"] > 0 for x in rows)
    r2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)       # resume: nothing new
    assert r2.returncode == 0
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 5
    dec = [json.loads(x) for x in (tmp_path / "seed0" / "calls.jsonl").read_text().splitlines()]
    assert any(d["kind"] == "decision" and d["replayed"] for d in dec)
