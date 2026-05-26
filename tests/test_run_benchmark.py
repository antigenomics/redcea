import pandas as pd

from benchmark.run_benchmark import standardize_redcea_assignments


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
