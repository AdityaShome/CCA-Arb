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

## Status

- M0: tau-bench snapshot/restore adapter and tests (done).
