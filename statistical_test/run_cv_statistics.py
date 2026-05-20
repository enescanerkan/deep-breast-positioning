#!/usr/bin/env python3
"""
10-fold CV statistical analysis (rebuttal / reviewer-aligned).

Dependencies: numpy, pandas, scipy; statsmodels optional (McNemar / Cochran Q fall back to scipy).

Outputs CSV tables under statistical_test/output/ by default.

Run (many Linux images ship only ``python3``):

    python3 statistical_test/run_cv_statistics.py

Classification: sensitivity / specificity match
``code/regression/main/utils/evaluation_utils.py::calculate_sensitivity_specificity`` and
``code/regression/main/evaluation.py`` summary lines. Binary labels: good=1, bad=0.

- **Sensitivity (code):** recall for the **bad** class — among true ``bad`` (0), fraction predicted ``bad``.

- **Specificity (code):** among true ``good`` (1), fraction predicted ``good`` (equivalently: recall for the good class).

Wilson / McNemar / Cochran use these same definitions.

Landmark (UNet vs RAUNet, mm_* + angular_distance): pooled test-set images across folds;
paired Wilcoxon, paired Cohen's d on paired differences **UNet − RAUNet** (errors in mm / degrees).
**Interpretation:** positive Cohen's d ⇒ higher **UNet** errors on average (UNet worse for that landmark).

Fold-level classification Cohen's d (Table 3): ``cohens_d_paired_a_minus_b`` for comparison ``A_vs_B`` is
mean(A_metric − B_metric) / SD(diff) across folds; for accuracy/sensitivity/specificity, **higher is better**, so
positive d ⇒ model **A** higher than **B** on that metric.

ResNeXt fold 1: predictions loaded from ``ResNeXt50_fold_1.csv`` (there is no ``ResNeXt50_fold_1_eval.json`` in this repo);
folds 2–10 use ``ResNeXt50_fold_{k}_eval.json``. See ``SUPPLEMENT_notes.txt`` in the output folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap, sem, t, wilcoxon

try:
    from statsmodels.stats.contingency_tables import cochrans_q as sm_cochrans_q
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
    from statsmodels.stats.proportion import proportion_confint

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

    def proportion_confint(count, nobs, alpha=0.05, method="wilson"):
        """Wilson interval without statsmodels (simple two-sided)."""
        if nobs <= 0:
            return (np.nan, np.nan)
        from math import sqrt

        z = stats.norm.ppf(1 - alpha / 2)
        phat = count / nobs
        denom = 1 + z**2 / nobs
        centre = phat + z**2 / (2 * nobs)
        adj = z * sqrt((phat * (1 - phat) + z**2 / (4 * nobs)) / nobs)
        low = (centre - adj) / denom
        high = (centre + adj) / denom
        return (max(0.0, low), min(1.0, high))


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_OUT = Path(__file__).resolve().parent / "output"

LANDMARK_COLS = [
    "mm_perpendicular",
    "mm_pec1",
    "mm_pec2",
    "mm_nipple",
    "angular_distance",
]

MODEL_ORDER = ["RAUNet", "UNet", "ResNeXt50"]
PRED_COLS = {"RAUNet": "pred_raunet", "UNet": "pred_unet", "ResNeXt50": "pred_resnext"}
PAIRWISE = [
    ("RAUNet", "UNet"),
    ("RAUNet", "ResNeXt50"),
    ("UNet", "ResNeXt50"),
]


def label_to_bin(s: pd.Series) -> np.ndarray:
    m = {"good": 1, "bad": 0}
    return s.astype(str).str.lower().map(m).to_numpy()


def calculate_sensitivity_specificity(predictions: np.ndarray, truths: np.ndarray):
    """
    Match ``evaluation_utils.calculate_sensitivity_specificity`` (regression + classification).

    Binary: good=1, bad=0. **Sensitivity** = recall for **bad** (0): TP_bad / (TP_bad + FN_bad) where
    TP_bad = (pred==0 & truth==0), FN_bad = (pred==1 & truth==0).
    **Specificity** = TN_good / (TN_good + FP_good) with TN_good = (pred==1 & truth==1),
    FP_good = (pred==0 & truth==1) — i.e. recall for **good** (1) among true good images.
    """
    p = predictions.astype(int)
    t = truths.astype(int)
    tp = np.sum((p == 0) & (t == 0))
    fn = np.sum((p == 1) & (t == 0))
    tn = np.sum((p == 1) & (t == 1))
    fp = np.sum((p == 0) & (t == 1))
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sens, spec


def fold_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = pred.astype(int)
    truth = truth.astype(int)
    acc = float(np.mean(pred == truth))
    sens, spec = calculate_sensitivity_specificity(pred, truth)
    return {"accuracy": acc, "sensitivity": float(sens), "specificity": float(spec)}


def mcnemar_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[float, np.ndarray]:
    """2x2 on correctness; return p-value and table [[a,b],[c,d]]."""
    a = int(np.sum(correct_a & correct_b))
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    d = int(np.sum(~correct_a & ~correct_b))
    table = np.array([[a, b], [c, d]])
    if _HAS_STATSMODELS:
        r = sm_mcnemar(table, exact=False, correction=True)
        return float(r.pvalue), table
    # continuity-corrected chi-square
    if b + c == 0:
        return 1.0, table
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    pval = float(stats.chi2.sf(stat, 1))
    return pval, table


def cochrans_q_pvalue(mat: np.ndarray) -> tuple[float, float]:
    """
    mat: (n, 3) binary — Cochran's Q (Wikipedia form).
    Returns (Q statistic, p-value).
    """
    x = np.asarray(mat, dtype=float)
    n, k = x.shape
    if k < 2:
        return np.nan, np.nan
    T = x.sum(axis=0)
    L = x.sum(axis=1)
    num = (k - 1) * (k * np.sum(T**2) - np.sum(T) ** 2)
    den = k * np.sum(L) - np.sum(L**2)
    if den <= 0:
        return np.nan, np.nan
    q = num / den
    if _HAS_STATSMODELS:
        try:
            r = sm_cochrans_q(x)
            return float(r.statistic), float(r.pvalue)
        except Exception:
            pass
    pval = float(stats.chi2.sf(q, k - 1))
    return float(q), pval


def wilcoxon_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, "no_data"
    if np.allclose(d, 0):
        return 1.0, "all_zero_diff"
    kw = dict(zero_method="wilcox", alternative="two-sided")
    try:
        res = wilcoxon(x, y, **kw, method="auto")
    except TypeError:
        res = wilcoxon(x, y, **kw)
    return float(res.pvalue), "ok"


def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    sd = np.std(d, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(np.mean(d) / sd)


def bootstrap_mean_ci(
    x: np.ndarray,
    n_resamples: int = 10_000,
    random_state: int = 0,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)

    def stat(*samples):
        return np.mean(samples[0])

    rng = np.random.default_rng(random_state)
    try:
        res = bootstrap(
            (x,),
            stat,
            vectorized=False,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=rng,
            method="percentile",
        )
        low, high = res.confidence_interval.low, res.confidence_interval.high
        return float(low), float(high)
    except Exception:
        # Fallback: percentile bootstrap on mean
        means = []
        n = x.size
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            means.append(float(np.mean(x[idx])))
        means.sort()
        lo_i = max(0, int((1 - confidence_level) / 2 * n_resamples))
        hi_i = min(n_resamples - 1, int((1 + confidence_level) / 2 * n_resamples))
        return means[lo_i], means[hi_i]


def load_resnext_fold(results_dir: Path, fold: int) -> pd.DataFrame:
    jpath = results_dir / f"ResNeXt50_fold_{fold}_eval.json"
    if jpath.is_file():
        with open(jpath, encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for r in data.get("per_image", []):
            rows.append(
                {
                    "study_uid": str(r["StudyInstanceUID"]),
                    "sop_uid": str(r["SOPInstanceUID"]),
                    "pred": int(r["prediction_label"]),
                }
            )
        return pd.DataFrame(rows)
    cpath = results_dir / "ResNeXt50_fold_1.csv"
    if fold == 1 and cpath.is_file():
        df = pd.read_csv(cpath)
        df = df.rename(
            columns={
                "StudyInstanceUID": "study_uid",
                "SOPInstanceUID": "sop_uid",
                "Prediction": "pred",
            }
        )
        if "Ground Truth" in df.columns:
            df["cls_gt"] = df["Ground Truth"].astype(int)
        df["study_uid"] = df["study_uid"].astype(str)
        df["sop_uid"] = df["sop_uid"].astype(str)
        return df[["study_uid", "sop_uid", "pred"] + (["cls_gt"] if "cls_gt" in df.columns else [])]
    raise FileNotFoundError(f"No ResNeXt eval for fold {fold}: missing {jpath} and fold-1 CSV")


def load_seg_per_image(results_dir: Path, name: str, fold: int) -> pd.DataFrame:
    path = results_dir / f"{name}_fold_{fold}_eval_per_image.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def merge_fold_classification(results_dir: Path, fold: int) -> pd.DataFrame:
    u = load_seg_per_image(results_dir, "UNet", fold)
    r = load_seg_per_image(results_dir, "RAUNet", fold)
    rx = load_resnext_fold(results_dir, fold)
    # ResNeXt eval JSON lists each image twice (400 rows / 200 images); preds agree — keep one row per key.
    rx = rx.drop_duplicates(subset=["study_uid", "sop_uid"], keep="first")

    u = u.rename(columns={"prediction_label": "pred_unet"})
    r = r.rename(columns={"prediction_label": "pred_raunet"})
    u["study_uid"] = u["study_uid"].astype(str)
    u["sop_uid"] = u["sop_uid"].astype(str)
    r["study_uid"] = r["study_uid"].astype(str)
    r["sop_uid"] = r["sop_uid"].astype(str)

    m = u.merge(
        r[["study_uid", "sop_uid", "pred_raunet"]],
        on=["study_uid", "sop_uid"],
        how="inner",
    )
    m = m.merge(rx, on=["study_uid", "sop_uid"], how="inner")
    m = m.rename(columns={"pred": "pred_resnext"})
    m["automated_truth"] = label_to_bin(m["automated_label"])
    m["manual_truth"] = label_to_bin(m["manual_label"])
    m["pred_unet"] = label_to_bin(m["pred_unet"])
    m["pred_raunet"] = label_to_bin(m["pred_raunet"])
    m["pred_resnext"] = m["pred_resnext"].astype(int)
    m["fold"] = fold
    return m


def merge_fold_landmark(results_dir: Path, fold: int) -> pd.DataFrame:
    u = load_seg_per_image(results_dir, "UNet", fold)
    r = load_seg_per_image(results_dir, "RAUNet", fold)
    for c in LANDMARK_COLS:
        if c not in u.columns or c not in r.columns:
            raise KeyError(f"Missing {c} in seg CSV")
    keys = ["study_uid", "sop_uid"]
    u = u[keys + LANDMARK_COLS].copy()
    r = r[keys + LANDMARK_COLS].copy()
    for df in (u, r):
        df["study_uid"] = df["study_uid"].astype(str)
        df["sop_uid"] = df["sop_uid"].astype(str)
    u = u.add_prefix("u_").rename(columns={"u_study_uid": "study_uid", "u_sop_uid": "sop_uid"})
    r = r.add_prefix("r_").rename(columns={"r_study_uid": "study_uid", "r_sop_uid": "sop_uid"})
    m = u.merge(r, on=["study_uid", "sop_uid"], how="inner")
    m["fold"] = fold
    return m


def descriptive_series(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "iqr_low": float(q1),
        "iqr_high": float(q3),
    }


def wilson_triplet(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    return proportion_confint(successes, n, alpha=0.05, method="wilson")


def run_landmark(pooled: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for col in LANDMARK_COLS:
        ucol, rcol = f"u_{col}", f"r_{col}"
        u = pooled[ucol].to_numpy(dtype=float)
        r = pooled[rcol].to_numpy(dtype=float)
        du = descriptive_series(u)
        dr = descriptive_series(r)
        diff = u - r
        p_w, _ = wilcoxon_safe(u, r)
        d_c = cohen_d_paired(u, r)
        b_lo, b_hi = bootstrap_mean_ci(diff, n_resamples=10_000, random_state=42)
        rows.append(
            {
                "landmark": col,
                "unet_mean": du["mean"],
                "unet_std": du["std"],
                "unet_median": du["median"],
                "unet_iqr_low": du["iqr_low"],
                "unet_iqr_high": du["iqr_high"],
                "raunet_mean": dr["mean"],
                "raunet_std": dr["std"],
                "raunet_median": dr["median"],
                "raunet_iqr_low": dr["iqr_low"],
                "raunet_iqr_high": dr["iqr_high"],
                "paired_diff_mean_unet_minus_raunet": float(np.nanmean(diff)),
                "bootstrap95ci_diff_mean_low": b_lo,
                "bootstrap95ci_diff_mean_high": b_hi,
                "wilcoxon_p_two_sided": p_w,
                "cohens_d_paired_unet_minus_raunet": d_c,
                "n_images": du["n"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table1_landmark_unet_vs_raunet.csv", index=False)
    return df


def per_image_classification(df: pd.DataFrame, ref: str, out_dir: Path) -> None:
    truth_col = "automated_truth" if ref == "automated" else "manual_truth"
    preds = {
        "RAUNet": df["pred_raunet"].to_numpy(),
        "UNet": df["pred_unet"].to_numpy(),
        "ResNeXt50": df["pred_resnext"].to_numpy(),
    }
    truth = df[truth_col].to_numpy().astype(int)
    n = truth.size

    rows_ci = []
    for model, pred in preds.items():
        correct = pred == truth
        acc = float(np.mean(correct))
        sens, spec = calculate_sensitivity_specificity(pred, truth)
        sens_n = int(np.sum(truth == 0))
        spec_n = int(np.sum(truth == 1))
        sens_succ = int(np.sum((truth == 0) & (pred == 0)))
        spec_succ = int(np.sum((truth == 1) & (pred == 1)))
        acc_lo, acc_hi = wilson_triplet(int(correct.sum()), n)
        s_lo, s_hi = wilson_triplet(sens_succ, sens_n)
        sp_lo, sp_hi = wilson_triplet(spec_succ, spec_n)
        rows_ci.append(
            {
                "reference": ref,
                "model": model,
                "n": n,
                "accuracy": acc,
                "accuracy_wilson_low": acc_lo,
                "accuracy_wilson_high": acc_hi,
                "sensitivity": sens,
                "sensitivity_wilson_low": s_lo,
                "sensitivity_wilson_high": s_hi,
                "specificity": spec,
                "specificity_wilson_low": sp_lo,
                "specificity_wilson_high": sp_hi,
            }
        )
    pd.DataFrame(rows_ci).to_csv(out_dir / f"table2_classification_per_image_{ref}.csv", index=False)

    correct_mat = np.column_stack([(preds[m] == truth).astype(int) for m in MODEL_ORDER])
    q_stat, q_p = cochrans_q_pvalue(correct_mat)

    mcnem_rows = []
    for a, b in PAIRWISE:
        ca = (preds[a] == truth).astype(bool)
        cb = (preds[b] == truth).astype(bool)
        p, tab = mcnemar_pvalue(ca, cb)
        mcnem_rows.append(
            {
                "reference": ref,
                "comparison": f"{a}_vs_{b}",
                "mcnemar_p": p,
                "n_both_correct": int(tab[0, 0]),
                "n_a_only_correct": int(tab[0, 1]),
                "n_b_only_correct": int(tab[1, 0]),
                "n_neither_correct": int(tab[1, 1]),
            }
        )
    mcnem_df = pd.DataFrame(mcnem_rows)
    mcnem_df.to_csv(out_dir / f"table4_mcnemar_{ref}.csv", index=False)

    pd.DataFrame([{"reference": ref, "cochran_q": q_stat, "p_value": q_p}]).to_csv(
        out_dir / f"table4_cochrans_q_{ref}.csv", index=False
    )


def fold_level_tables(
    fold_rows: list[dict],
    ref: str,
    out_dir: Path,
) -> None:
    fr = pd.DataFrame(fold_rows)
    fr = fr[fr["reference"] == ref]
    metrics = ["accuracy", "sensitivity", "specificity"]
    t_low, t_high = t.ppf(0.025, 9), t.ppf(0.975, 9)

    desc = []
    wilcox = []
    cohen = []
    for model in MODEL_ORDER:
        for met in metrics:
            x = fr.loc[fr["model"] == model, met].to_numpy(dtype=float)
            if x.size != 10:
                raise ValueError(f"Expected 10 folds for {model} {met}, got {x.size}")
            m = float(np.mean(x))
            s = float(np.std(x, ddof=1))
            se = float(sem(x, ddof=1))
            ci_lo = m + t_low * se
            ci_hi = m + t_high * se
            desc.append(
                {
                    "reference": ref,
                    "model": model,
                    "metric": met,
                    "mean": m,
                    "std": s,
                    "tci95_low": ci_lo,
                    "tci95_high": ci_hi,
                }
            )

    for a, b in PAIRWISE:
        for met in metrics:
            sub = fr[["fold", "model", met]].drop_duplicates()
            piv = sub.pivot(index="fold", columns="model", values=met).sort_index()
            if a not in piv.columns or b not in piv.columns:
                raise ValueError(f"Missing model column in fold pivot for {met}: {piv.columns.tolist()}")
            xa = piv[a].to_numpy(dtype=float)
            xb = piv[b].to_numpy(dtype=float)
            p_w, _ = wilcoxon_safe(xa, xb)
            d = cohen_d_paired(xa, xb)
            wilcox.append(
                {
                    "reference": ref,
                    "comparison": f"{a}_vs_{b}",
                    "metric": met,
                    "wilcoxon_p_two_sided": p_w,
                }
            )
            cohen.append(
                {
                    "reference": ref,
                    "comparison": f"{a}_vs_{b}",
                    "metric": met,
                    "cohens_d_paired_a_minus_b": d,
                }
            )

    pd.DataFrame(desc).to_csv(out_dir / f"table3_fold_descriptives_{ref}.csv", index=False)
    pd.DataFrame(wilcox).to_csv(out_dir / f"table3_fold_wilcoxon_{ref}.csv", index=False)
    pd.DataFrame(cohen).to_csv(out_dir / f"table3_fold_cohens_d_{ref}.csv", index=False)


def write_supplement_notes(out_dir: Path, results_dir: Path) -> None:
    """
    Reviewer-facing notes: metric definitions, Cohen's d direction, ResNeXt fold-1 provenance,
    and pooled RAUNet numbers from this run (if table2 CSVs exist) for manuscript cross-check.
    """
    lines = [
        "SUPPLEMENT — statistical_test pipeline notes",
        "============================================",
        "",
        "1) Sensitivity / specificity (same as training evaluation code)",
        "----------------------------------------------------------------",
        "Implemented in:",
        "  - code/regression/main/utils/evaluation_utils.py :: calculate_sensitivity_specificity",
        "  - code/classification/utils/metrics.py :: calculate_sensitivity_specificity (same logic)",
        "  - code/regression/main/evaluation.py (summary_stats strings use these).",
        "",
        "Binary labels: good = 1, bad = 0 (mapped from good/bad strings in segmentation CSVs).",
        "",
        "Sensitivity (as named in code): recall for the BAD class — among images with truth == bad (0),",
        "  fraction with prediction == bad (0).",
        "",
        "Specificity (as named in code): among images with truth == good (1), fraction with prediction == good (1).",
        "  (Equivalently: recall for the good / adequate class.)",
        "",
        "The statistical script uses this same function for Wilson intervals, fold tables, and pooled table2.",
        "",
        "2) Manuscript number cross-check (RAUNet, pooled test N across folds)",
        "---------------------------------------------------------------------",
        "If the manuscript reports e.g. 87.53% sensitivity and 88.62% specificity, compare explicitly to:",
        "  - which reference line (PNL automated_label vs radiologist manual_label),",
        "  - pooled vs single-fold summary_stats,",
        "  - and the values in table2_classification_per_image_*.csv from this run (below).",
        "",
        "With disease / target class = bad (0) and non-disease = good (1), these formulas match the usual",
        "binary definitions: sensitivity = P(pred=bad | true=bad), specificity = P(pred=good | true=good).",
        "",
    ]

    auto_path = out_dir / "table2_classification_per_image_automated.csv"
    man_path = out_dir / "table2_classification_per_image_manual.csv"
    for label, path in (("automated (PNL)", auto_path), ("manual (radiologist)", man_path)):
        if path.is_file():
            df = pd.read_csv(path)
            row = df[df["model"] == "RAUNet"]
            if len(row) == 1:
                r = row.iloc[0]
                lines.append(
                    f"  RAUNet pooled ({label}), n={int(r['n'])}: "
                    f"sensitivity={100*r['sensitivity']:.4f}%, specificity={100*r['specificity']:.4f}%, "
                    f"accuracy={100*r['accuracy']:.4f}%"
                )
    lines.extend(
        [
            "",
            "3) ResNeXt fold 1 — different artifact than folds 2–10",
            "-------------------------------------------------------",
            "Fold 1: ResNeXt predictions are read from "
            f"{results_dir / 'ResNeXt50_fold_1.csv'} (no ResNeXt50_fold_1_eval.json in this repo).",
            f"Folds 2–10: read from {results_dir}/ResNeXt50_fold_<k>_eval.json (per_image list).",
            "JSON exports for folds 2–10 contain duplicate rows per (Study,SOP); the script keeps one row per key.",
            "If fold-level plots show fold 1 as an isolated point for ResNeXt-only contrasts, alternate file format",
            "or preprocessing for fold 1 may contribute; interpret alongside this note.",
            "",
            "4) Cohen's d — sign / direction",
            "-------------------------------",
            "Landmark table1: column cohens_d_paired_unet_minus_raunet = mean(UNet_error - RAUNet_error) / SD(diffs).",
            "  POSITIVE => UNet has larger errors on average (worse landmark localization than RAUNet).",
            "  NEGATIVE => RAUNet has larger errors (UNet better).",
            "",
            "Classification table3: cohens_d_paired_a_minus_b for comparison A_vs_B uses the same formula on",
            "fold-level accuracy, sensitivity, or specificity. Higher metric is better — positive d => model A",
            "higher than model B on that metric.",
            "",
        ]
    )

    (out_dir / "SUPPLEMENT_notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(1, 11)))
    args = ap.parse_args()
    results_dir: Path = args.results_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _HAS_STATSMODELS:
        print("Note: statsmodels not installed; Wilson / McNemar / Cochran use scipy fallbacks.", file=sys.stderr)

    # --- Landmark ---
    land_parts = []
    for fold in args.folds:
        try:
            land_parts.append(merge_fold_landmark(results_dir, fold))
        except Exception as e:
            print(f"Landmark fold {fold} skipped: {e}", file=sys.stderr)
    if not land_parts:
        print("No landmark data loaded.", file=sys.stderr)
    else:
        pooled_land = pd.concat(land_parts, ignore_index=True)
        dup = pooled_land.duplicated(subset=["study_uid", "sop_uid"]).sum()
        if dup:
            print(f"Warning: {dup} duplicate (study,sop) rows in pooled landmark data.", file=sys.stderr)
        run_landmark(pooled_land, out_dir)

    # --- Classification pooled + fold metrics ---
    cls_parts = []
    fold_metric_rows = []
    supp_rows = []

    for fold in args.folds:
        try:
            m = merge_fold_classification(results_dir, fold)
        except Exception as e:
            print(f"Classification fold {fold} skipped: {e}", file=sys.stderr)
            continue
        cls_parts.append(m)
        for ref in ("automated", "manual"):
            truth_col = "automated_truth" if ref == "automated" else "manual_truth"
            truth = m[truth_col].to_numpy().astype(int)
            for model in MODEL_ORDER:
                pred = m[PRED_COLS[model]].to_numpy().astype(int)
                met = fold_metrics(pred, truth)
                fold_metric_rows.append(
                    {
                        "fold": fold,
                        "reference": ref,
                        "model": model,
                        "accuracy": met["accuracy"],
                        "sensitivity": met["sensitivity"],
                        "specificity": met["specificity"],
                    }
                )
                supp_rows.append(
                    {
                        "fold": fold,
                        "reference": ref,
                        "model": model,
                        **met,
                    }
                )

    if cls_parts:
        pooled_cls = pd.concat(cls_parts, ignore_index=True)
        dup = pooled_cls.duplicated(subset=["study_uid", "sop_uid"]).sum()
        if dup:
            print(f"Warning: {dup} duplicate (study,sop) in pooled classification.", file=sys.stderr)
        for ref in ("automated", "manual"):
            per_image_classification(pooled_cls, ref, out_dir)
        fr_df = pd.DataFrame(fold_metric_rows)
        pd.DataFrame(supp_rows).to_csv(out_dir / "supplement_fold_by_fold_metrics.csv", index=False)
        for ref in ("automated", "manual"):
            fold_level_tables(fr_df.to_dict("records"), ref, out_dir)

    write_supplement_notes(out_dir, results_dir)

    print(f"Wrote tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
