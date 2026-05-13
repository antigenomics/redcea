from pathlib import Path

import numpy as np
import pandas as pd

from redcea.analysis.io import get_enrichment_column_names
from redcea.utils.paths import resolve_embedding_file
from redcea.utils.stats import _fdr_bh, add_binom_pvalues, add_log_fold_change, add_z_binom_pvalues


def test_fdr_bh_returns_monotone_qvalues_in_original_order():
    qvals = _fdr_bh([0.04, 0.001, 0.02, 0.5])

    assert qvals.shape == (4,)
    assert np.all((qvals >= 0) & (qvals <= 1))
    assert qvals[1] <= qvals[2] <= qvals[0] <= qvals[3]


def test_resolve_embedding_file_builds_default_path(tmp_path: Path):
    resolved = resolve_embedding_file(None, tmp_path, "sample1", "background")
    assert resolved == tmp_path / "sample1_background_embeddings.parquet"


def test_resolve_embedding_file_raises_when_must_exist(tmp_path: Path):
    missing = tmp_path / "missing.parquet"
    try:
        resolve_embedding_file(str(missing), tmp_path, "x", "sample", must_exist=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for missing embedding file")


def test_enrichment_helpers_add_expected_columns():
    summary = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "sample": [10, 3, 1],
            "background": [1, 3, 10],
        }
    )

    summary = add_z_binom_pvalues(summary, total_sample=20, total_background=20)
    summary = add_log_fold_change(summary, total_sample=20, total_background=20)

    assert "enrichment_pvalue_zbinom" in summary.columns
    assert "enrichment_fdr_zbinom" in summary.columns
    assert "log_fold_change" in summary.columns
    assert np.all((summary["enrichment_fdr_zbinom"] >= 0) & (summary["enrichment_fdr_zbinom"] <= 1))
    assert summary.loc[summary["cluster_id"] == 1, "log_fold_change"].iat[0] > 0
    assert summary.loc[summary["cluster_id"] == 3, "log_fold_change"].iat[0] < 0


def test_binom_helper_and_enrichment_column_names():
    summary = pd.DataFrame(
        {
            "cluster_id": [1, 2],
            "sample": [10, 1],
            "background": [2, 8],
        }
    )

    summary = add_binom_pvalues(summary, total_sample=20, total_background=20)
    pvalue_col, fdr_col = get_enrichment_column_names("binom")

    assert pvalue_col == "enrichment_pvalue_binom"
    assert fdr_col == "enrichment_fdr_binom"
    assert pvalue_col in summary.columns
    assert fdr_col in summary.columns
    assert np.all((summary[fdr_col] >= 0) & (summary[fdr_col] <= 1))
