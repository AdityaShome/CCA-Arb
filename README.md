# CCA-Arb

Counterfactual credit assignment for proposal arbitration: replay rejected proposals from an environment checkpoint and use the paired outcome difference to learn per-agent, per-decision-type reliability.

## Setup

```
python -m venv .venv
.venv/Scripts/activate            # Windows; use .venv/bin/activate on Linux
pip install -r requirements.txt
git clone --depth 1 https://github.com/sierra-research/tau-bench third_party/tau-bench
PYTHONUTF8=1 pip install --no-deps -e third_party/tau-bench
pytest tests -q
```

`PYTHONUTF8=1` is needed on Windows because tau-bench's `setup.py` reads its README with the default codec.

## Layout
`cca/` library (envs, agents, canon, dtype, replay, arbiters, trigger, metrics), `scripts/run_experiment.py` (resumable, config-driven), `scripts/analyze.py`, `scripts/make_figures.py`, `configs/`, `docs/` (baseline interpretations and design decisions), `notebooks/colab_setup.ipynb`.

Dry run without a GPU (mock agents, real tau-bench state; not a research result):
`python scripts/run_experiment.py configs/dev_mock.yaml`

## Status
- M0 done: snapshot/restore adapter, verified.
- M1: agents, client, logging written and unit-tested with fakes; not yet run against real models.
- M2-M5 code written and tested against mocks and a synthetic simulator; no real-model results yet.
- M6 (SWE-bench) and M7 (experiments) not started.
- `drf`, `rr`, `bda` are unverified interpretations; see `docs/baselines.md`.
