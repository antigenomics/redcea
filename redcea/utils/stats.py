from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import betabinom, binom, fisher_exact, norm


def _fdr_bh(pvals: list[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if p.size == 0:
        return p

    order = np.argsort(p)
    ranked = p[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    qvals = np.empty_like(adjusted)
    qvals[order] = adjusted
    return qvals


def add_fisher_pvalues(summary: pd.DataFrame, total_sample: int, total_background: int) -> pd.DataFrame:
    pvals = []
    for _, row in summary.iterrows():
        a = row.get("sample", 0)
        c = row.get("background", 0)
        b = total_sample - a
        d = total_background - c
        _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
        pvals.append(pval)

    summary["enrichment_pvalue_fisher"] = pvals
    summary["enrichment_fdr_fisher"] = _fdr_bh(pvals)
    return summary


def add_binom_pvalues(summary: pd.DataFrame, total_sample: int, total_background: int) -> pd.DataFrame:
    pvals = []
    for _, row in summary.iterrows():
        a = row.get("sample", 0)
        b = row.get("background", 0)
        p = b / total_background
        pval = min(binom.sf(k=a, n=total_sample, p=p), binom.cdf(k=a, n=total_sample, p=p))
        pvals.append(pval)

    summary["enrichment_pvalue_binom"] = pvals
    summary["enrichment_fdr_binom"] = _fdr_bh(pvals)
    return summary


def add_z_binom_pvalues(summary: pd.DataFrame, total_sample: int, total_background: int) -> pd.DataFrame:
    pvals = []
    for _, row in summary.iterrows():
        a = row.get("sample", 0)
        b = row.get("background", 0)
        z = (total_background * a - total_sample * b) / np.sqrt(total_sample * (b + 1e-6) * (total_background - b))
        pvals.append(2 * (1 - norm.cdf(np.abs(z))))

    summary["enrichment_pvalue_zbinom"] = pvals
    summary["enrichment_fdr_zbinom"] = _fdr_bh(pvals)
    return summary


def add_beta_binom_pvalues(summary: pd.DataFrame, total_sample: float, total_background: float) -> pd.DataFrame:
    pvals = []
    total_sample = float(total_sample)
    total_background = float(total_background)
    eps = np.finfo(float).eps

    for _, row in summary.iterrows():
        k_sample = float(row.get("sample_count_sum", 0.0))
        k_background = float(row.get("background_count_sum", 0.0))

        if total_sample <= 0 or total_background <= 0:
            pvals.append(1.0)
            continue

        bg_usage = k_background / total_background
        a = float(np.clip(bg_usage * total_background, eps, None))
        b = float(np.clip((1.0 - bg_usage) * total_background, eps, None))

        observed = int(round(k_sample))
        n = int(round(total_sample))
        left_tail = float(betabinom.cdf(observed, n=n, a=a, b=b))
        right_tail = float(betabinom.sf(observed - 1, n=n, a=a, b=b))
        pvals.append(float(min(1.0, 2.0 * min(left_tail, right_tail))))

    summary["enrichment_pvalue_betabinom_expansion"] = pvals
    summary["enrichment_fdr_betabinom_expansion"] = _fdr_bh(pvals)
    return summary


def add_log_fold_change(summary: pd.DataFrame, total_sample: int, total_background: int) -> pd.DataFrame:
    log_fc = []
    for _, row in summary.iterrows():
        a = row.get("sample", 0)
        b = row.get("background", 0)
        fold_enrichment = ((a + 1e-9) / total_sample) / ((b + 1e-9) / total_background)
        log_fc.append(np.log10(fold_enrichment))
    summary["log_fold_change"] = log_fc
    return summary


def add_count_frequency_columns(summary: pd.DataFrame, total_sample: float, total_background: float) -> pd.DataFrame:
    if "sample_count_sum" not in summary.columns or "background_count_sum" not in summary.columns:
        return summary

    total_sample = float(total_sample)
    total_background = float(total_background)
    summary["sample_frequency"] = summary["sample_count_sum"].astype(float) / total_sample if total_sample > 0 else 0.0
    summary["background_frequency"] = (
        summary["background_count_sum"].astype(float) / total_background if total_background > 0 else 0.0
    )
    summary["expansion_log_fold_change"] = np.log10(
        (summary["sample_count_sum"].astype(float) + 1e-9) / max(total_sample, 1e-9)
    ) - np.log10((summary["background_count_sum"].astype(float) + 1e-9) / max(total_background, 1e-9))
    return summary

