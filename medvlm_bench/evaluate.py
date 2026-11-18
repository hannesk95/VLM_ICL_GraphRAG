"""Aggregate prediction files into a results table with CIs and paired comparisons.

Threshold policy: AUROC/AUPRC/Brier/ECE are threshold-free (or use raw probabilities). For
sensitivity/specificity/balanced accuracy we report both the default 0.5 threshold and a threshold
selected on the VALIDATION split (Youden/balanced accuracy) - never tuned on test.
Stochastic conditions (several seeds): metrics are computed per seed (mean, sd reported) and the
bootstrap CI is computed on the seed-averaged probability.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci, mcnemar_exact, paired_bootstrap_delta, point_metrics, youden_threshold


def _load(pred_dir: Path, split: str) -> pd.DataFrame:
    fs = sorted((pred_dir / split).glob("*.csv")) if (pred_dir / split).exists() else []
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _pooled(df: pd.DataFrame, cond: str) -> pd.DataFrame:
    d = df[df.condition == cond]
    return d.groupby("pid").agg(y=("y", "first"), p=("p_pos", "mean")).reset_index().sort_values("pid")


def evaluate(out_dir: str | Path, ref: str = "zs_v0", n_boot: int = 1000):
    out = Path(out_dir)
    pdir = out / "predictions"
    test, val = _load(pdir, "test"), _load(pdir, "val")
    if test.empty:
        raise SystemExit("No test predictions found.")
    rows, thr_of = [], {}
    for cond in sorted(test.condition.unique()):
        t = _pooled(test, cond)
        thr = 0.5
        if not val.empty and cond in set(val.condition):
            v = _pooled(val, cond)
            if v.y.nunique() == 2:
                thr = youden_threshold(v.y.values, v.p.values)
        thr_of[cond] = thr
        seeds = test[test.condition == cond]
        per_seed = [point_metrics(g.y.values, g.p_pos.values, 0.5)["auroc"] for _, g in seeds.groupby("seed")]
        pm, pv = point_metrics(t.y.values, t.p.values, 0.5), point_metrics(t.y.values, t.p.values, thr)
        ci = bootstrap_ci(t.y.values, t.p.values, 0.5, n_boot)
        sd = seeds[seeds.condition == cond]
        rows.append(dict(
            condition=cond, n=len(t), n_pos=int(t.y.sum()), n_seeds=seeds.seed.nunique(),
            auroc=pm["auroc"], auroc_lo=ci["auroc"][0], auroc_hi=ci["auroc"][1],
            auroc_seed_sd=float(np.std(per_seed)) if len(per_seed) > 1 else np.nan,
            auprc=pm["auprc"], brier=pm["brier"], ece=pm["ece"],
            bal_acc_at_0p5=pm["bal_acc"], thr_val=thr, bal_acc_at_valthr=pv["bal_acc"],
            sens_at_valthr=pv["sens"], spec_at_valthr=pv["spec"], f1_at_valthr=pv["f1"],
            order_sensitivity=float(np.nanmean(np.abs(sd.p_pos_o0 - sd.p_pos_o1))) if sd.p_pos_o1.notna().any() else np.nan,
            option_mass=float(sd.option_mass.mean()), mean_images=float(sd.n_images.mean()),
        ))
    res = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    res.to_csv(out / "results_table.csv", index=False)

    paired = []
    if ref in thr_of:
        r = _pooled(test, ref)
        for cond in thr_of:
            if cond == ref:
                continue
            t = _pooled(test, cond)
            m = r.merge(t, on="pid", suffixes=("_r", "_c"))
            if m.empty or m.y_r.nunique() < 2:
                continue
            d, lo, hi, p = paired_bootstrap_delta(m.y_r.values, m.p_c.values, m.p_r.values, "auroc", n_boot=n_boot)
            mc = mcnemar_exact(m.y_r.values, m.p_c.values, m.p_r.values, thr_of[cond], thr_of[ref])
            paired.append(dict(condition=cond, vs=ref, delta_auroc=d, ci_lo=lo, ci_hi=hi, p_boot=p,
                               mcnemar_p=mc["p"], cond_only_correct=mc["a_only_correct"], ref_only_correct=mc["b_only_correct"]))
    pd.DataFrame(paired).to_csv(out / "paired_vs_ref.csv", index=False)
    with open(out / "results.md", "w") as f:
        f.write("## Results (test split)\n\n" + res.round(3).to_markdown(index=False) + "\n")
        if paired:
            f.write(f"\n## Paired comparison vs `{ref}`\n\n" + pd.DataFrame(paired).round(3).to_markdown(index=False) + "\n")
    return res, pd.DataFrame(paired)


def _thr_metrics(y, p, thr_per_patient):
    """Sensitivity / specificity / balanced accuracy when every patient has its own (fold-specific) threshold."""
    y, pred = np.asarray(y), np.asarray(p) >= np.asarray(thr_per_patient)
    sens = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return sens, spec, (sens + spec) / 2


def evaluate_cv(cv_dir: str | Path, ref: str = "zs_v0", n_boot: int = 1000):
    """Stratified K-fold CV: <cv_dir>/fold0, fold1, ... each produced with `--n-folds N --fold k`.

    Every patient is a test patient in exactly one fold, so the test predictions of all folds are pooled into
    one out-of-fold prediction per patient (pooled AUROC with patient-level bootstrap CI), and the spread of
    the per-fold AUROC shows split-to-split variability. The decision threshold of each fold is chosen on that
    fold's own validation patients (never on its test fold) and applied to that fold's test patients.
    Only conditions that were run on ALL folds are reported.
    """
    root = Path(cv_dir)
    fdirs = sorted((d for d in root.glob("fold*") if d.is_dir()), key=lambda d: int(d.name[4:]))
    if not fdirs:
        raise SystemExit(f"No fold*/ directories in {root}")
    tests, vals = [], []
    for i, d in enumerate(fdirs):
        t, v = _load(d / "predictions", "test"), _load(d / "predictions", "val")
        if t.empty:
            print(f"warning: no test predictions in {d.name}; skipped")
            continue
        t["fold"] = i
        v["fold"] = i
        tests.append(t); vals.append(v)
    test, val = pd.concat(tests, ignore_index=True), pd.concat(vals, ignore_index=True)
    n_f = test.fold.nunique()
    conds = [c for c in sorted(test.condition.unique()) if test[test.condition == c].fold.nunique() == n_f]
    skipped = sorted(set(test.condition) - set(conds))
    if skipped:
        print("warning: skipped (not run on all folds):", ", ".join(skipped))

    def pooled(df, cond):  # one row per patient (mean over seeds), with the fold it was tested in
        d = df[df.condition == cond]
        return d.groupby("pid").agg(y=("y", "first"), p=("p_pos", "mean"), fold=("fold", "first")).reset_index().sort_values("pid")

    rows, pooled_of = [], {}
    for cond in conds:
        t = pooled(test, cond); pooled_of[cond] = t
        thr_fold = {}
        for f in sorted(t.fold.unique()):
            v = pooled(val[val.fold == f], cond) if cond in set(val[val.fold == f].condition) else pd.DataFrame()
            thr_fold[f] = youden_threshold(v.y.values, v.p.values) if (not v.empty and v.y.nunique() == 2) else 0.5
        thr = t.fold.map(thr_fold).values
        fold_auc = [point_metrics(g.y.values, g.p.values, 0.5)["auroc"] for _, g in t.groupby("fold") if g.y.nunique() == 2]
        pm = point_metrics(t.y.values, t.p.values, 0.5)
        ci = bootstrap_ci(t.y.values, t.p.values, 0.5, n_boot)
        sens, spec, bal = _thr_metrics(t.y.values, t.p.values, thr)
        sd = test[test.condition == cond]
        rows.append(dict(
            condition=cond, n=len(t), n_pos=int(t.y.sum()), n_folds=len(fold_auc),
            auroc_pooled=pm["auroc"], auroc_lo=ci["auroc"][0], auroc_hi=ci["auroc"][1],
            auroc_fold_mean=float(np.mean(fold_auc)), auroc_fold_sd=float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else np.nan,
            auprc=pm["auprc"], brier=pm["brier"], ece=pm["ece"], bal_acc_at_0p5=pm["bal_acc"],
            sens_at_foldthr=sens, spec_at_foldthr=spec, bal_acc_at_foldthr=bal,
            order_sensitivity=float(np.nanmean(np.abs(sd.p_pos_o0 - sd.p_pos_o1))) if sd.p_pos_o1.notna().any() else np.nan,
            option_mass=float(sd.option_mass.mean()), mean_images=float(sd.n_images.mean()),
        ))
    res = pd.DataFrame(rows).sort_values("auroc_pooled", ascending=False)
    res.to_csv(root / "results_table_cv.csv", index=False)

    paired = []
    if ref in pooled_of:
        r = pooled_of[ref]
        for cond in conds:
            if cond == ref:
                continue
            m = r.merge(pooled_of[cond], on="pid", suffixes=("_r", "_c"))
            if m.empty or m.y_r.nunique() < 2:
                continue
            d, lo, hi, p = paired_bootstrap_delta(m.y_r.values, m.p_c.values, m.p_r.values, "auroc", n_boot=n_boot)
            paired.append(dict(condition=cond, vs=ref, delta_auroc=d, ci_lo=lo, ci_hi=hi, p_boot=p))
    pd.DataFrame(paired).to_csv(root / "paired_vs_ref_cv.csv", index=False)
    with open(root / "results_cv.md", "w") as f:
        f.write(f"## Results ({n_f}-fold stratified CV, out-of-fold predictions pooled)\n\n" + res.round(3).to_markdown(index=False) + "\n")
        if paired:
            f.write(f"\n## Paired comparison vs `{ref}` (pooled out-of-fold)\n\n" + pd.DataFrame(paired).round(3).to_markdown(index=False) + "\n")
    return res, pd.DataFrame(paired)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="zs_v0")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--cv", action="store_true", help="--out is the parent of fold0/, fold1/, ... (K-fold CV)")
    a = ap.parse_args(argv)
    if a.cv:
        res, _ = evaluate_cv(a.out, a.ref, a.n_boot)
        print(res[["condition", "n", "auroc_pooled", "auroc_lo", "auroc_hi", "auroc_fold_mean", "auroc_fold_sd",
                   "bal_acc_at_foldthr", "brier", "option_mass"]].round(3).to_string(index=False))
        return
    res, paired = evaluate(a.out, a.ref, a.n_boot)
    print(res[["condition", "n", "auroc", "auroc_lo", "auroc_hi", "bal_acc_at_valthr", "brier", "ece", "option_mass"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
