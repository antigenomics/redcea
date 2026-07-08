from __future__ import annotations

import pandas as pd

from benchmark.evaluation import _compute_vdjdb_metrics_for_frame


def test_vdjdb_f1_counts_any_non_noise_cluster_as_recovered() -> None:
    assignments = pd.DataFrame(
        [
            {
                "run_id": "demo_run",
                "epitope": "GLCTLVAML",
                "method": "dbscan",
                "parameter_json": "{}",
                "truth_label": "positive",
                "cluster_id": 0,
                "is_noise": False,
            },
            {
                "run_id": "demo_run",
                "epitope": "GLCTLVAML",
                "method": "dbscan",
                "parameter_json": "{}",
                "truth_label": "negative",
                "cluster_id": 0,
                "is_noise": False,
            },
            {
                "run_id": "demo_run",
                "epitope": "GLCTLVAML",
                "method": "dbscan",
                "parameter_json": "{}",
                "truth_label": "positive",
                "cluster_id": -1,
                "is_noise": True,
            },
        ]
    )

    metrics = _compute_vdjdb_metrics_for_frame(assignments, metadata_row={})

    assert metrics is not None
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
