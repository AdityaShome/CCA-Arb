"""Statistics (GUIDE 7): paired bootstrap CI, Wilcoxon signed-rank, Holm correction."""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap_ci(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """95% CI of mean(a - b) resampling tasks. Returns (mean_diff, lo, hi)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_p(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def compare_arbiters(scores: dict[str, list[float]], ref: str, seed: int = 0) -> list[dict]:
    """Each arbiter vs `ref` on per-task scores (aligned lists): diff, CI, Wilcoxon, Holm-p."""
    others = [k for k in scores if k != ref]
    raw = {k: wilcoxon_p(scores[k], scores[ref]) for k in others}
    adj = holm(raw)
    rows = []
    for k in others:
        m, lo, hi = paired_bootstrap_ci(scores[k], scores[ref], seed=seed)
        rows.append(dict(arbiter=k, ref=ref, mean_diff=m, ci_lo=lo, ci_hi=hi,
                         p_raw=raw[k], p_holm=adj[k], n=len(scores[k])))
    return rows
