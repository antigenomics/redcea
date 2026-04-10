from __future__ import annotations

import pandas as pd


def has_vdjdb_match(x, vdjdb, threshold: int = 1) -> bool:
    return len(vdjdb.get_matching_clonotypes(x, threshold=threshold)) > 0


def get_cluster_matches_vdjdb(df: pd.DataFrame, cluster_id, vdjdb, threshold: int = 1) -> bool:
    cluster_df = df[df.cluster_id == cluster_id]
    matches = cluster_df.cdr3aa_beta.apply(lambda x: has_vdjdb_match(x, vdjdb, threshold=threshold))
    return bool(matches.any())


__all__ = [
    "get_cluster_matches_vdjdb",
    "has_vdjdb_match",
]
