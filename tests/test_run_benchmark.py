import pandas as pd

from benchmark.run_benchmark import build_execution_manifest, standardize_redcea_assignments


def test_standardize_redcea_assignments_accepts_legacy_beta_columns():
    cluster_df = pd.DataFrame(
        {
            "clone_id": ["s_0", "b_0"],
            "cluster_id": [2, -1],
            "cdr3aa_beta": ["CASSLGQDTQYF", "CASSIRSSYEQYF"],
            "v_beta": ["TRBV7-2*01", "TRBV6-5*01"],
            "j_beta": ["TRBJ2-3*01", "TRBJ2-7*01"],
            "source": ["sample", "background"],
        }
    )
    truth_table = pd.DataFrame(columns=["epitope_aa", "chain", "cdr3", "v_gene", "j_gene", "truth_label"])

    assignments = standardize_redcea_assignments(
        cluster_df,
        run_id="demo_vdjdb_vdbscan_grid_0001",
        dataset="demo",
        dataset_mode="vdjdb",
        method="vdbscan",
        parameter_json="{}",
        epitope="GILGFVFTL",
        donor_id=None,
        truth_table=truth_table,
    )

    assert {"junction_aa", "v_call", "j_call", "locus"} <= set(assignments.columns)
    assert assignments["junction_aa"].tolist() == ["CASSLGQDTQYF", "CASSIRSSYEQYF"]
    assert assignments["v_call"].tolist() == ["TRBV7-2*01", "TRBV6-5*01"]
    assert assignments["j_call"].tolist() == ["TRBJ2-3*01", "TRBJ2-7*01"]
    assert assignments["locus"].tolist() == ["beta", "beta"]


def test_build_execution_manifest_filters_to_yfv_p1_p2_lowres_grid(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame(
        [
            {"dataset": "vdjdb_glc", "dataset_mode": "vdjdb", "dataset_key": "GLC", "epitope": "GLCTLVAML", "donor_id": None},
            {"dataset": "yfv_repertoires", "dataset_mode": "yfv", "dataset_key": "P1_F1", "epitope": None, "donor_id": "P1_F1"},
            {"dataset": "yfv_repertoires", "dataset_mode": "yfv", "dataset_key": "P2_F1", "epitope": None, "donor_id": "P2_F1"},
            {"dataset": "yfv_repertoires", "dataset_mode": "yfv", "dataset_key": "Q1_F1", "epitope": None, "donor_id": "Q1_F1"},
        ]
    ).to_csv(processed_dir / "benchmark_dataset_manifest.tsv", sep="\t", index=False)

    manifest = build_execution_manifest(
        processed_dir,
        grid_size="yfv_vdbscan_leiden_lowres",
        dataset_mode_filter="yfv",
        yfv_donor_ids=["P1_F1", "P2_F1"],
    )

    assert manifest["dataset_mode"].tolist().count("yfv") == len(manifest)
    assert set(manifest["donor_id"]) == {"P1_F1", "P2_F1"}
    assert manifest["method"].nunique() == 1
    assert manifest["method"].iloc[0] == "vdbscan_leiden"
    assert len(manifest) == 36
