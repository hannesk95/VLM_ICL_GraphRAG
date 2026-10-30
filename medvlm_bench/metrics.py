"""Evaluation metrics, patient-level bootstrap CIs and paired comparisons."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def youden_threshold(y, p) -> float:
    """Threshold maximising balanced accuracy (select on VALIDATION data only)."""
    cand = np.unique(np.concatenate([[0.5], p]))
    best, bt = -1.0, 0.5
    for t in cand:
        ba = balanced_accuracy_score(y, (p >= t).astype(int))
        if ba > best + 1e-12:
            best, bt = ba, float(t)
    return bt


def point_metrics(y, p, thr: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp, tn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 0)).sum()
    P, N = (y == 1).sum(), (y == 0).sum()
    both = P > 0 and N > 0
    return {
        "auroc": roc_auc_score(y, p) if both else np.nan,
        "auprc": average_precision_score(y, p) if P > 0 else np.nan,
        "bal_acc": balanced_accuracy_score(y, pred) if both else np.nan,
        "sens": tp / P if P else np.nan,
        "spec": tn / N if N else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
    }


def bootstrap_ci(y, p, thr: float = 0.5, n_boot: int = 1000, seed: int = 0, metrics=("auroc", "bal_acc", "auprc")):
    """Percentile 95% CIs from a bootstrap over patients (rows must be one-per-patient)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    acc = {m: [] for m in metrics}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        pm = point_metrics(y[i], p[i], thr)
        for m in metrics:
            acc[m].append(pm[m])
    return {m: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for m, v in acc.items()}


def paired_bootstrap_delta(y, p_a, p_b, metric: str = "auroc", thr_a=0.5, thr_b=0.5, n_boot: int = 2000, seed: int = 0):
    """delta = metric(A) - metric(B) on the same patients; returns (delta, ci_lo, ci_hi, two-sided p)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    n = len(y)
    obs = point_metrics(y, p_a, thr_a)[metric] - point_metrics(y, p_b, thr_b)[metric]
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        d.append(point_metrics(y[i], p_a[i], thr_a)[metric] - point_metrics(y[i], p_b[i], thr_b)[metric])
    d = np.asarray(d)
    pval = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), pval


def mcnemar_exact(y, p_a, p_b, thr_a=0.5, thr_b=0.5) -> dict:
    y = np.asarray(y)
    ca = (np.asarray(p_a) >= thr_a).astype(int) == y
    cb = (np.asarray(p_b) >= thr_b).astype(int) == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pval = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "p": pval}
