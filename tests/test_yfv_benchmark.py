from __future__ import annotations

import math

import pandas as pd

from benchmark.yfv_benchmark import (
    build_match_key,
    compute_overlap_fraction,
    compute_requested_lfc_mass_shift,
    is_enriched_cluster_value,
)


def test_llw_exact_cdr3_matching_uses_exact_match_only():
    llw_reference = pd.DataFrame(
        {
            "cdr3": ["CASSLLWQETQYF"],
            "v_gene": ["TRBV7-2"],
            "j_gene": ["TRBJ2-3"],
        }
    )
    clonotypes = pd.DataFrame(
        {
            "cdr3": ["CASSLLWQETQYF", "CASSLLWQETQFF"],
            "v_gene": ["TRBV7-2", "TRBV7-2"],
            "j_gene": ["TRBJ2-3", "TRBJ2-3"],
        }
    )

    llw_keys = set(build_match_key(llw_reference, "cdr3"))
    clonotype_keys = build_match_key(clonotypes, "cdr3")

    assert clonotype_keys.iloc[0] in llw_keys
    assert clonotype_keys.iloc[1] not in llw_keys


def test_overlap_fraction_uses_min_denominator():
    left = {"A", "B", "C"}
    right = {"B", "C", "D", "E"}

    assert math.isclose(compute_overlap_fraction(left, right), 2.0 / 3.0)


def test_requested_lfc_mass_shift_never_returns_nan():
    cluster_summary = pd.DataFrame(
        {
            "qvalue": [0.001, 0.02, 1.0],
            "log2fc": [1.5, -0.3, 0.0],
        }
    )

    value = compute_requested_lfc_mass_shift(cluster_summary)

    assert not math.isnan(value)
    assert 0.0 <= value <= 1.0


def test_enriched_threshold_requires_qvalue_and_positive_log2fc():
    assert is_enriched_cluster_value(0.049, 0.1)
    assert not is_enriched_cluster_value(0.05, 0.1)
    assert not is_enriched_cluster_value(0.049, 0.0)
