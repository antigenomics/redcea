from __future__ import annotations

from collections import defaultdict

import networkx as nx
import pandas as pd


def compute_cluster_summary(cluster_df: pd.DataFrame, sample_ids) -> pd.DataFrame:
    """Summarize sample/background membership for each non-noise cluster."""
    if "source" not in cluster_df.columns:
        sample_ids_set = set(sample_ids)
        cluster_df = cluster_df.copy()
        cluster_df["source"] = cluster_df["clone_id"].isin(sample_ids_set).map({True: "sample", False: "background"})

    summary = (
        cluster_df.groupby("cluster_id")["source"]
        .value_counts()
        .unstack(fill_value=0)
        .rename_axis(index="cluster_id", columns=None)
        .reset_index()
    )
    summary["cluster_size"] = summary.get("sample", 0) + summary.get("background", 0)
    return summary[summary.cluster_id != -1]


def merge_clusters_by_shared_cdr3(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    cdr3_col: str = "cdr3aa_beta",
    merged_col: str = "merged_cluster_id",
) -> tuple[pd.DataFrame, dict]:
    """Merge clusters that share at least one identical CDR3 sequence."""
    cdr3_to_clusters = defaultdict(set)
    for _, row in df.iterrows():
        cdr3_to_clusters[row[cdr3_col]].add(row[cluster_col])

    graph = nx.Graph()
    graph.add_nodes_from(df[cluster_col].unique())
    for clusters in cdr3_to_clusters.values():
        clusters = list(clusters)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                graph.add_edge(clusters[i], clusters[j])

    cluster_mapping = {}
    for new_id, component in enumerate(nx.connected_components(graph)):
        for cluster_id in component:
            cluster_mapping[cluster_id] = new_id

    df_out = df.copy()
    df_out[merged_col] = df_out[cluster_col].map(cluster_mapping)
    return df_out, cluster_mapping


def get_cluster_usage(
    df: pd.DataFrame,
    cluster_idx,
    clonotype_to_patients,
    merged_col: str = "merged_cluster_id",
    print_info: bool = False,
) -> int:
    """Count unique patients contributing clonotypes to a merged cluster."""
    del print_info

    all_patients = set()
    for clono_idx in df[df[merged_col] == cluster_idx].index:
        patients = clonotype_to_patients[clono_idx]
        all_patients |= patients
    return len(all_patients)


__all__ = [
    "compute_cluster_summary",
    "get_cluster_usage",
    "merge_clusters_by_shared_cdr3",
]
