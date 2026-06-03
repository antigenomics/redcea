from __future__ import annotations

import numpy as np
import pandas as pd


AUXILIARY_CLUSTER_METRIC_COLUMNS = [
    "sample_usage",
    "background_usage",
    "sample_fraction_in_cluster",
    "log2fc_smooth",
    "fc_smooth",
    "minus_log10_fdr",
    "support_log",
    "enrichment_support_score",
    "enrichment_stat_score",
    "internal_edges_n",
    "internal_mreach_q90",
    "cohesion",
    "external_edges_n",
    "external_mreach_q10",
    "separation",
    "density_validity_full",
    "density_validity_enriched_only",
    "density_validity",
    "has_density_validity",
    "is_sample_knn_closed",
    "geometry_score",
    "log2_odds",
    "odds_ratio",
    "dbcv_with_odds",
    "redcea_auxiliary_validity",
    "is_good_candidate",
    "is_good_candidate_relaxed",
]

_GEOMETRY_COLUMNS = [
    "internal_edges_n",
    "internal_mreach_q90",
    "cohesion",
    "external_edges_n",
    "external_mreach_q10",
    "separation",
    "density_validity_full",
    "density_validity_enriched_only",
    "density_validity",
    "geometry_score",
]

_ALPHA = 0.5
_CORE_K = 15
_FDR_SCORE_CAP = 50.0
_FDR_THRESHOLD = 0.05
_MIN_SAMPLE_SUPPORT = 5
_MIN_DENSITY_VALIDITY = 0.0
_NOISE_CLUSTER_ID = -1


def append_auxiliary_cluster_metrics(
    summary_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    total_sample: int,
    total_background: int,
    sample_knn_indices: np.ndarray | None = None,
    sample_knn_distances: np.ndarray | None = None,
) -> pd.DataFrame:
    """Append lightweight enrichment and geometry metrics to a cluster summary."""
    if total_sample <= 0 or total_background <= 0:
        raise ValueError("Both sample and background totals must be positive.")

    prepared_cluster_df = _prepare_cluster_df(cluster_df)
    summary = _drop_existing_metric_columns(summary_df)
    summary = _append_enrichment_metrics(
        summary,
        total_sample=total_sample,
        total_background=total_background,
    )

    geometry_df = _compute_geometry_metrics(
        cluster_df=prepared_cluster_df,
        summary_df=summary,
        sample_knn_indices=sample_knn_indices,
        sample_knn_distances=sample_knn_distances,
    )
    summary = summary.merge(geometry_df, on="cluster_id", how="left")
    summary = _append_geometry_flags(summary)

    sample_values = summary["sample"].to_numpy(dtype=np.float64, copy=False)
    background_values = summary["background"].to_numpy(dtype=np.float64, copy=False)
    density_validity = summary["density_validity"].to_numpy(dtype=np.float64, copy=False)

    log2_odds, odds_ratio, dbcv_with_odds = _compute_odds_metrics(
        sample=sample_values,
        background=background_values,
        total_sample=total_sample,
        total_background=total_background,
        density_validity=density_validity,
    )
    summary["log2_odds"] = log2_odds
    summary["odds_ratio"] = odds_ratio
    summary["dbcv_with_odds"] = dbcv_with_odds

    summary = _append_candidate_metrics(summary)
    summary = _clear_noise_cluster_metrics(summary)
    return summary


def _drop_existing_metric_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        column
        for column in AUXILIARY_CLUSTER_METRIC_COLUMNS
        if column in summary_df.columns
    ]
    return summary_df.drop(columns=columns_to_drop).copy()


def _append_enrichment_metrics(
    summary_df: pd.DataFrame,
    *,
    total_sample: int,
    total_background: int,
) -> pd.DataFrame:
    summary = summary_df.copy()
    sample_count = summary["sample"].astype("float64")
    background_count = summary["background"].astype("float64")
    cluster_size = summary["cluster_size"].astype("float64")

    summary["sample_usage"] = sample_count / float(total_sample)
    summary["background_usage"] = background_count / float(total_background)
    summary["sample_fraction_in_cluster"] = np.where(cluster_size > 0, sample_count / cluster_size, np.nan)

    summary["log2fc_smooth"] = (
        np.log2((sample_count + _ALPHA) / float(total_sample))
        - np.log2((background_count + _ALPHA) / float(total_background))
    )
    summary["fc_smooth"] = np.power(2.0, summary["log2fc_smooth"])

    fdr_values = summary["enrichment_fdr_zbinom"].astype("float64")
    safe_fdr_values = np.maximum(fdr_values, 1e-300)
    summary["minus_log10_fdr"] = -np.log10(safe_fdr_values)
    summary["support_log"] = np.log1p(sample_count)

    positive_log2fc = np.maximum(summary["log2fc_smooth"], 0.0)
    capped_fdr_score = np.minimum(summary["minus_log10_fdr"], _FDR_SCORE_CAP)
    summary["enrichment_support_score"] = positive_log2fc * summary["support_log"]
    summary["enrichment_stat_score"] = (
        positive_log2fc
        * summary["support_log"]
        * capped_fdr_score
    )
    return summary


def _compute_geometry_metrics(
    *,
    cluster_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sample_knn_indices: np.ndarray | None,
    sample_knn_distances: np.ndarray | None,
) -> pd.DataFrame:
    cluster_ids = summary_df["cluster_id"].to_numpy(dtype=np.int64, copy=False)
    geometry = _empty_geometry_frame(cluster_ids)
    sample_df = cluster_df.loc[cluster_df["source"] == "sample"].reset_index(drop=True)
    if sample_knn_indices is None or sample_knn_distances is None or sample_df.empty:
        return geometry
    if sample_knn_indices.shape != sample_knn_distances.shape:
        raise ValueError(
            f"Sample kNN indices/distances shape mismatch: {sample_knn_indices.shape} vs {sample_knn_distances.shape}"
        )
    if sample_knn_indices.shape[0] != len(sample_df):
        raise ValueError(
            f"Sample kNN row count {sample_knn_indices.shape[0]} does not match sample cluster row count {len(sample_df)}."
        )

    point_cluster_ids = sample_df["cluster_id"].to_numpy(dtype=np.int64, copy=False)
    core_dist = _compute_core_distances(
        knn_indices=sample_knn_indices,
        knn_distances=sample_knn_distances,
        core_k=_CORE_K,
    )

    n_points = sample_knn_indices.shape[0]
    row_ids = np.arange(n_points, dtype=np.int64)[:, None]
    valid_mask = sample_knn_indices != row_ids

    src_idx = np.repeat(np.arange(n_points, dtype=np.int64), sample_knn_indices.shape[1])
    tgt_idx = sample_knn_indices.reshape(-1).astype(np.int64, copy=False)
    edge_dist = sample_knn_distances.reshape(-1).astype(np.float64, copy=False)
    valid = valid_mask.reshape(-1) & np.isfinite(edge_dist) & (tgt_idx >= 0) & (tgt_idx < n_points)

    src_idx = src_idx[valid]
    tgt_idx = tgt_idx[valid]
    edge_dist = edge_dist[valid]

    src_cluster = point_cluster_ids[src_idx]
    tgt_cluster = point_cluster_ids[tgt_idx]
    non_noise = src_cluster != _NOISE_CLUSTER_ID
    src_idx = src_idx[non_noise]
    tgt_idx = tgt_idx[non_noise]
    edge_dist = edge_dist[non_noise]
    src_cluster = src_cluster[non_noise]
    tgt_cluster = tgt_cluster[non_noise]

    edge_mreach = np.maximum.reduce([core_dist[src_idx], core_dist[tgt_idx], edge_dist])

    geometry = _populate_geometry_stats(
        geometry_df=geometry,
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
        edge_mreach=edge_mreach,
    )

    enriched_cluster_ids = _get_enriched_cluster_ids(summary_df)
    enriched_only = _compute_enriched_only_density_validity(
        cluster_ids=cluster_ids,
        enriched_cluster_ids=enriched_cluster_ids,
        src_cluster=src_cluster,
        tgt_cluster=tgt_cluster,
        edge_mreach=edge_mreach,
    )
    geometry["density_validity_full"] = geometry["density_validity"]
    geometry["density_validity_enriched_only"] = enriched_only
    return geometry


def _populate_geometry_stats(
    *,
    geometry_df: pd.DataFrame,
    src_cluster: np.ndarray,
    tgt_cluster: np.ndarray,
    edge_mreach: np.ndarray,
) -> pd.DataFrame:
    geometry = geometry_df.copy()
    internal_mask = src_cluster == tgt_cluster
    external_mask = src_cluster != tgt_cluster

    internal_stats = _summarize_cluster_mreach(
        cluster_ids=src_cluster[internal_mask],
        mreach_values=edge_mreach[internal_mask],
        count_column="internal_edges_n",
        quantile_column="internal_mreach_q90",
        quantile=0.90,
    )
    external_stats = _summarize_cluster_mreach(
        cluster_ids=src_cluster[external_mask],
        mreach_values=edge_mreach[external_mask],
        count_column="external_edges_n",
        quantile_column="external_mreach_q10",
        quantile=0.10,
    )

    if not internal_stats.empty:
        geometry = _merge_geometry_stats(
            geometry,
            internal_stats,
            count_column="internal_edges_n",
            quantile_column="internal_mreach_q90",
        )

    if not external_stats.empty:
        geometry = _merge_geometry_stats(
            geometry,
            external_stats,
            count_column="external_edges_n",
            quantile_column="external_mreach_q10",
        )

    geometry["cohesion"] = geometry["internal_mreach_q90"]
    geometry["separation"] = geometry["external_mreach_q10"]
    geometry["density_validity"] = _compute_density_validity(
        cohesion=geometry["cohesion"],
        separation=geometry["separation"],
    )
    geometry["geometry_score"] = geometry["density_validity"].clip(lower=0)
    return geometry


def _compute_density_validity(
    *,
    cohesion: pd.Series,
    separation: pd.Series,
) -> pd.Series:
    cohesion = cohesion.astype("float64")
    separation = separation.astype("float64")
    denom = np.maximum(cohesion, separation)
    density_validity = pd.Series(np.nan, index=cohesion.index, dtype="float64")

    finite_mask = cohesion.notna() & separation.notna()
    positive_denom = finite_mask & (denom > 0)
    zero_denom = finite_mask & (denom == 0)

    density_validity.loc[positive_denom] = (
        (separation.loc[positive_denom] - cohesion.loc[positive_denom]) / denom.loc[positive_denom]
    )
    density_validity.loc[zero_denom] = 0.0
    return density_validity


def _get_enriched_cluster_ids(summary_df: pd.DataFrame) -> pd.Index:
    enriched_mask = (
        (summary_df["log_fold_change"].astype("float64") > 0)
        & (summary_df["enrichment_fdr_zbinom"].astype("float64") < _FDR_THRESHOLD)
        & (summary_df["cluster_id"].astype("int64") != _NOISE_CLUSTER_ID)
    )
    return pd.Index(summary_df.loc[enriched_mask, "cluster_id"].astype("int64").unique(), dtype="int64")


def _compute_enriched_only_density_validity(
    *,
    cluster_ids: np.ndarray,
    enriched_cluster_ids: pd.Index,
    src_cluster: np.ndarray,
    tgt_cluster: np.ndarray,
    edge_mreach: np.ndarray,
) -> pd.Series:
    enriched_only = pd.Series(np.nan, index=np.arange(len(cluster_ids)), dtype="float64")
    if len(enriched_cluster_ids) < 2:
        return enriched_only

    enriched_set = set(enriched_cluster_ids.tolist())
    src_enriched = np.isin(src_cluster, enriched_cluster_ids.to_numpy(dtype=np.int64, copy=False))
    tgt_enriched = np.isin(tgt_cluster, enriched_cluster_ids.to_numpy(dtype=np.int64, copy=False))
    relevant_edges = src_enriched & tgt_enriched
    if not np.any(relevant_edges):
        return enriched_only

    enriched_geometry = _empty_geometry_frame(enriched_cluster_ids.to_numpy(dtype=np.int64, copy=False))
    enriched_geometry = _populate_geometry_stats(
        geometry_df=enriched_geometry,
        src_cluster=src_cluster[relevant_edges],
        tgt_cluster=tgt_cluster[relevant_edges],
        edge_mreach=edge_mreach[relevant_edges],
    )
    valid_count = int(enriched_geometry["density_validity"].notna().sum())
    if valid_count < 2:
        return enriched_only

    density_by_cluster = enriched_geometry.set_index("cluster_id")["density_validity"]
    for idx, cluster_id in enumerate(cluster_ids):
        if int(cluster_id) in enriched_set:
            enriched_only.iloc[idx] = density_by_cluster.get(int(cluster_id), np.nan)
    return enriched_only


def _summarize_cluster_mreach(
    *,
    cluster_ids: np.ndarray,
    mreach_values: np.ndarray,
    count_column: str,
    quantile_column: str,
    quantile: float,
) -> pd.DataFrame:
    if cluster_ids.size == 0:
        return pd.DataFrame(columns=["cluster_id", count_column, quantile_column])

    edges_df = pd.DataFrame(
        {
            "cluster_id": cluster_ids.astype(np.int64, copy=False),
            "mreach": mreach_values.astype(np.float64, copy=False),
        }
    )
    return (
        edges_df.groupby("cluster_id", sort=False)["mreach"]
        .agg(
            **{
                count_column: "size",
                quantile_column: lambda values: float(values.quantile(quantile)),
            }
        )
        .reset_index()
    )


def _merge_geometry_stats(
    geometry_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    *,
    count_column: str,
    quantile_column: str,
) -> pd.DataFrame:
    merged = geometry_df.merge(stats_df, on="cluster_id", how="left", suffixes=("", "__new"))
    merged[count_column] = merged[f"{count_column}__new"].fillna(merged[count_column]).astype("int64")
    merged[quantile_column] = merged[f"{quantile_column}__new"].fillna(merged[quantile_column])
    return merged.drop(columns=[f"{count_column}__new", f"{quantile_column}__new"])


def _compute_core_distances(
    *,
    knn_indices: np.ndarray,
    knn_distances: np.ndarray,
    core_k: int,
) -> np.ndarray:
    n_points, width = knn_indices.shape
    if width == 0:
        return np.full(n_points, np.nan, dtype=np.float64)

    row_ids = np.arange(n_points, dtype=np.int64)[:, None]
    valid_mask = knn_indices != row_ids
    filtered = np.where(valid_mask, knn_distances, np.inf)
    filtered_sorted = np.sort(filtered, axis=1)
    valid_counts = valid_mask.sum(axis=1)
    core_pos = np.minimum(np.maximum(valid_counts, 1), core_k) - 1
    core_dist = filtered_sorted[np.arange(n_points), core_pos]
    core_dist[valid_counts == 0] = np.nan
    return core_dist.astype("float64", copy=False)


def _empty_geometry_frame(cluster_ids: np.ndarray) -> pd.DataFrame:
    unique_cluster_ids = pd.Index(pd.unique(cluster_ids), dtype="int64")
    geometry = pd.DataFrame({"cluster_id": unique_cluster_ids})
    geometry["internal_edges_n"] = 0
    geometry["internal_mreach_q90"] = np.nan
    geometry["cohesion"] = np.nan
    geometry["external_edges_n"] = 0
    geometry["external_mreach_q10"] = np.nan
    geometry["separation"] = np.nan
    geometry["density_validity_full"] = np.nan
    geometry["density_validity_enriched_only"] = np.nan
    geometry["density_validity"] = np.nan
    geometry["geometry_score"] = np.nan
    return geometry


def _compute_odds_metrics(
    *,
    sample: np.ndarray,
    background: np.ndarray,
    total_sample: int,
    total_background: int,
    density_validity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = np.asarray(sample, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    density_validity = np.asarray(density_validity, dtype=np.float64)

    sample_outside = float(total_sample) - sample
    background_outside = float(total_background) - background
    log2_odds = (
        np.log2((sample + _ALPHA) / (sample_outside + _ALPHA))
        - np.log2((background + _ALPHA) / (background_outside + _ALPHA))
    )
    odds_ratio = np.power(2.0, log2_odds)

    positive_log2_odds = np.where(sample > 0, np.maximum(log2_odds, 0.0), 0.0)
    geometry_part = np.where(
        np.isfinite(density_validity),
        np.maximum(density_validity, 0.0),
        np.nan,
    )
    dbcv_with_odds = geometry_part * positive_log2_odds * np.log1p(sample)
    return log2_odds, odds_ratio, dbcv_with_odds


def _append_geometry_flags(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary = summary_df.copy()
    internal_edge_counts = summary["internal_edges_n"].to_numpy(dtype=np.int64, copy=False)
    external_edge_counts = summary["external_edges_n"].to_numpy(dtype=np.int64, copy=False)
    density_validity = summary["density_validity"].to_numpy(dtype=np.float64, copy=False)

    summary["has_density_validity"] = (
        (internal_edge_counts > 0)
        & (external_edge_counts > 0)
        & np.isfinite(density_validity)
    )
    summary["is_sample_knn_closed"] = (
        (internal_edge_counts > 0)
        & (external_edge_counts == 0)
    )
    return summary


def _append_candidate_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary = summary_df.copy()
    positive_log2fc = np.maximum(summary["log2fc_smooth"], 0.0)
    passes_enrichment_cutoff = (
        (summary["enrichment_fdr_zbinom"] < _FDR_THRESHOLD)
        & (summary["log2fc_smooth"] > 0)
        & (summary["sample"] >= _MIN_SAMPLE_SUPPORT)
    )
    passes_density_cutoff = summary["density_validity"] > _MIN_DENSITY_VALIDITY

    summary["redcea_auxiliary_validity"] = (
        summary["geometry_score"] * positive_log2fc * summary["support_log"]
    )
    summary["is_good_candidate"] = (
        passes_enrichment_cutoff
        & passes_density_cutoff
    )
    summary["is_good_candidate_relaxed"] = (
        passes_enrichment_cutoff
        & (passes_density_cutoff | summary["is_sample_knn_closed"])
    )
    return summary


def _clear_noise_cluster_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary = summary_df.copy()
    noise_mask = summary["cluster_id"] == _NOISE_CLUSTER_ID
    if not noise_mask.any():
        return summary

    columns_to_blank = _GEOMETRY_COLUMNS + [
        "has_density_validity",
        "is_sample_knn_closed",
        "dbcv_with_odds",
        "redcea_auxiliary_validity",
    ]
    summary.loc[noise_mask, columns_to_blank] = np.nan
    summary.loc[noise_mask, "is_good_candidate"] = False
    summary.loc[noise_mask, "is_good_candidate_relaxed"] = False
    return summary


def _prepare_cluster_df(cluster_df: pd.DataFrame) -> pd.DataFrame:
    prepared = cluster_df.copy()
    if "source" not in prepared.columns:
        clone_ids = prepared["clone_id"].astype(str)
        prepared["source"] = np.where(
            clone_ids.str.startswith("s_"),
            "sample",
            np.where(clone_ids.str.startswith("b_"), "background", None),
        )
    if prepared["source"].isna().any():
        missing = int(prepared["source"].isna().sum())
        raise ValueError(f"Could not infer source for {missing} cluster rows.")
    return prepared


__all__ = [
    "AUXILIARY_CLUSTER_METRIC_COLUMNS",
    "append_auxiliary_cluster_metrics",
]
