"""Figures from run logs (GUIDE section 8). Reads only logged decisions; draws nothing it cannot cite.

usage: python scripts/make_figures.py runs/a runs/b ... --out figures [--persuasive m_pers]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cca.metrics.cost import read_jsonl            # noqa: E402
from cca.metrics.regret import acceptance_share, cumulative_regret  # noqa: E402


def decisions_of(d: Path) -> list[dict]:
    out = []
    for s in sorted(d.glob("seed*/calls.jsonl")):
        out += [r for r in read_jsonl(s) if r.get("kind") == "decision"]
    return out


def label_of(d: Path) -> str:
    rows = [r for r in read_jsonl(d / "results.jsonl") if not r.get("failed")]
    return f"{rows[0]['arbiter']}+{rows[0]['trigger']}" if rows else d.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--persuasive", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [(label_of(Path(r)), decisions_of(Path(r))) for r in a.runs]

    fig, ax = plt.subplots(figsize=(6, 4))                                  # Fig 2
    for lab, dec in runs:
        c = cumulative_regret(dec)
        if len(c):
            ax.plot(np.arange(1, len(c) + 1), c, label=lab)
    ax.set_xlabel("replayed decisions"); ax.set_ylabel("cumulative regret"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "regret_curve.png", dpi=150); plt.close(fig)

    if a.persuasive:                                                        # Fig 3
        fig, ax = plt.subplots(figsize=(6, 4))
        for lab, dec in runs:
            s = acceptance_share(dec, a.persuasive)
            if len(s):
                ax.plot(s, label=lab)
        ax.set_xlabel("disagreement index"); ax.set_ylabel(f"rolling acceptance share of {a.persuasive}")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "persuasion_drift.png", dpi=150); plt.close(fig)

    for lab, dec in runs:                                                   # Fig 5 + Fig 6 data
        agents = sorted({x for d in dec for x in d["agents"]})
        types = ["plan", "tool_choice", "code_edit"]
        M = np.full((len(agents), len(types)), np.nan)
        for i, ag in enumerate(agents):
            for j, t in enumerate(types):
                ys = [d["ys"][d["agents"].index(ag)] for d in dec
                      if d.get("replayed") and d["decision_type"] == t and ag in d["agents"]]
                if ys:
                    M[i, j] = np.mean(ys)
        fig, ax = plt.subplots(figsize=(5, 0.6 * len(agents) + 1.5))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(types)), types); ax.set_yticks(range(len(agents)), agents)
        fig.colorbar(im, label="mean replayed Y"); ax.set_title(lab)
        fig.tight_layout(); fig.savefig(out / f"reliability_{lab.replace('+', '_')}.png", dpi=150)
        plt.close(fig)
    print("wrote figures to", out)


if __name__ == "__main__":
    main()
