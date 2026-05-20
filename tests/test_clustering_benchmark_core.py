from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark import (
    ClusteringBenchmarkRunner,
    compute_cross_donor_overlap,
    compute_known_yfv_recovery,
    compute_vdjdb_metrics,
    compute_yfv_cluster_enrichment,
    summarize_yfv_enrichment,
)


def _toy_embedding_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clonotype_id": [f"c{i}" for i in range(6)],
            "cdr3": ["CASSPGQ", "CASSPGQ", "CASSLGQ", "CATSDGQ", "CATSDGQ", "CATSRGQ"],
            "v_gene": ["TRBV1"] * 6,
            "j_gene": ["TRBJ1"] * 6,
            "chain": ["TRB"] * 6,
            "truth_label": ["positive", "positive", "negative", "positive", "negative", "unlabeled"],
            "emb_0": [0.0, 0.01, 0.02, 5.0, 5.02, 5.04],
            "emb_1": [0.0, 0.01, 0.03, 5.0, 5.01, 5.03],
        }
    )


def test_runner_writes_standardized_outputs(tmp_path: Path):
    runner = ClusteringBenchmarkRunner(
        assignments_dir=tmp_path / "assignments",
        metadata_path=tmp_path / "run_metadata.tsv",
    )
    result = runner.run(
        _toy_embedding_frame(),
        dataset="vdjdb_glc",
        dataset_mode="vdjdb",
        method="dbscan",
        parameters={
            "min_samples": 2,
            "epsilon_strategy": "percentile",
            "percentile": 90,
            "distance_metric": "euclidean",
        },
        epitope="GLC",
        run_id="toy_dbscan",
    )
    assert result.metadata["status"] == "success"
    assert (tmp_path / "assignments" / "toy_dbscan.parquet").exists()
    assert set(["run_id", "dataset", "dataset_mode", "cluster_id", "is_noise"]).issubset(
        pd.read_parquet(tmp_path / "assignments" / "toy_dbscan.parquet").columns
    )


def test_compute_vdjdb_metrics_returns_expected_columns():
    assignments = pd.DataFrame(
        {
            "run_id": ["r1"] * 5,
            "epitope": ["GLC"] * 5,
            "method": ["dbscan"] * 5,
            "parameter_json": ['{"min_samples":2}'] * 5,
            "truth_label": ["positive", "positive", "negative", "negative", "unlabeled"],
            "cluster_id": [0, 0, 0, -1, 1],
            "is_noise": [False, False, False, True, False],
        }
    )
    run_metadata = pd.DataFrame(
        {
            "run_id": ["r1"],
            "n_clusters": [2],
            "noise_fraction": [0.2],
            "runtime_seconds": [0.01],
        }
    )
    metrics = compute_vdjdb_metrics(assignments, run_metadata)
    assert {"precision", "recall", "f1", "weighted_cluster_purity"}.issubset(metrics.columns)
    assert metrics.iloc[0]["precision"] > 0


def test_yfv_enrichment_recovery_and_overlap_pipeline():
    assignments = pd.DataFrame(
        {
            "run_id": ["runA"] * 4 + ["runB"] * 4,
            "donor_id": ["S1"] * 4 + ["S2"] * 4,
            "method": ["vdbscan_length"] * 8,
            "parameter_json": ['{"min_samples":2}'] * 8,
            "cluster_id": [0, 0, 1, -1, 0, 0, 1, -1],
            "is_noise": [False, False, False, True, False, False, False, True],
            "sample_label": ["sample", "background", "sample", "background"] * 2,
            "cdr3": ["AAA", "BBB", "YFV1", "CCC", "AAA", "DDD", "YFV1", "EEE"],
            "v_gene": ["TRBV1"] * 8,
            "j_gene": ["TRBJ1"] * 8,
            "chain": ["TRB"] * 8,
        }
    )
    enrichment = compute_yfv_cluster_enrichment(assignments)
    summary = summarize_yfv_enrichment(enrichment)
    known = pd.DataFrame(
        {
            "known_yfv_clonotype_id": ["k1"],
            "cdr3": ["YFV1"],
            "v_gene": ["TRBV1"],
            "j_gene": ["TRBJ1"],
            "epitope": ["LLWNGPMAV"],
            "source": ["VDJdb"],
        }
    )
    recovery = compute_known_yfv_recovery(assignments, enrichment, known)
    overlap = compute_cross_donor_overlap(assignments, enrichment)
    assert not enrichment.empty
    assert not summary.empty
    assert not recovery.empty
    assert not overlap.empty
