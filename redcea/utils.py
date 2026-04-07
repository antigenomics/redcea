from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, fisher_exact, norm

from tcremp import get_resource_path
from tcremp.utils import (
    configure_logging,
    generate_output_prefix,
    get_representations_df,
    load_analysis_repertoire,
    load_prototype_repertoire,
    log_memory_usage,
    prepare_output_path,
    resolve_input_file,
    resolve_prototype_file as _resolve_tcremp_prototype_file,
    subsample_repertoire,
)


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


def resolve_prototype_file(path: str | None, chain: str | None = None) -> str:
    if path:
        return str(Path(path).resolve())
    if chain is not None:
        return _resolve_tcremp_prototype_file(path, chain)
    return str(Path(get_resource_path("tcremp_prototypes_olga.tsv")).resolve())


def resolve_embedding_file(
    custom_path: str | None,
    output_path: str | Path,
    prefix: str,
    tag: str,
    must_exist: bool = False,
) -> Path:
    path = Path(custom_path) if custom_path else Path(output_path) / f"{prefix}_{tag}_embeddings.parquet"
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Embedding file for '{tag}' not found: {path}")
    return path


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


def add_log_fold_change(summary: pd.DataFrame, total_sample: int, total_background: int) -> pd.DataFrame:
    log_fc = []
    for _, row in summary.iterrows():
        a = row.get("sample", 0)
        b = row.get("background", 0)
        fold_enrichment = ((a + 1e-9) / total_sample) / ((b + 1e-9) / total_background)
        log_fc.append(np.log10(fold_enrichment))
    summary["log_fold_change"] = log_fc
    return summary


__all__ = [
    "add_binom_pvalues",
    "add_fisher_pvalues",
    "add_log_fold_change",
    "add_z_binom_pvalues",
    "configure_logging",
    "generate_output_prefix",
    "get_representations_df",
    "load_analysis_repertoire",
    "load_prototype_repertoire",
    "log_memory_usage",
    "prepare_output_path",
    "resolve_embedding_file",
    "resolve_input_file",
    "resolve_prototype_file",
    "subsample_repertoire",
]
