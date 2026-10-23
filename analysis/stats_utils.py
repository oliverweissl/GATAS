"""Shared statistics for the RQ notebooks: bootstrap / Wilson CIs, paired tests and effect sizes (matched by sentence_id)."""
import numpy as np
from scipy.stats import wilcoxon, binomtest

_RNG = np.random.default_rng(0)


def bootstrap_ci(x, stat=np.mean, n_boot: int = 5000, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    idx = _RNG.integers(0, len(x), (n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def paired_values(records: dict, a: str, b: str, getter):
    """getter(record) for methods a and b, restricted to the sentences both cover."""
    va = {r["sentence_id"]: getter(r) for r in records[a]}
    vb = {r["sentence_id"]: getter(r) for r in records[b]}
    ids = sorted(set(va) & set(vb))
    return np.array([va[i] for i in ids], dtype=float), np.array([vb[i] for i in ids], dtype=float)


def paired_compare(x, y, binary: bool = False, alternative: str = "two-sided") -> dict:
    """
    x - y per sentence. Continuous: Wilcoxon signed-rank, paired Cohen's d_z (signed) and mean difference, both with
    bootstrap 95% CI. Binary: exact McNemar (binomial on discordant pairs), difference of proportions with bootstrap CI,
    and the discordant-pair odds ratio (x-only / y-only successes).
    """
    diff = x - y
    out = {"n": len(x), "mean x": x.mean(), "mean y": y.mean(), "diff": diff.mean()}
    out["diff CI"] = bootstrap_ci(diff)
    if binary:
        n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        out["p"] = binomtest(n10, n10 + n01, 0.5, alternative=alternative).pvalue if n10 + n01 else 1.0
        out["effect"] = n10 / n01 if n01 else np.inf
        out["effect name"] = "odds ratio"
        out["effect CI"] = (np.nan, np.nan)
    else:
        out["p"] = 1.0 if np.allclose(x, y) else wilcoxon(x, y, alternative=alternative).pvalue
        sd = diff.std(ddof=1)
        d = diff.mean() / sd if sd > 0 else np.nan
        out["effect"] = d
        out["effect name"] = "d_z"
        out["effect CI"] = bootstrap_ci(diff, stat=lambda a, axis: a.mean(axis=axis) / np.maximum(a.std(axis=axis, ddof=1), 1e-12))
    return out


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


