from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binom


def _fdr_bh(pvals):
    p = np.asarray(list(pvals), dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _compute_vdjdb_recovered_mask(labeled_all: pd.DataFrame) -> pd.Series:
    """Mark labeled clonotypes as recovered when they belong to any non-noise cluster."""
    return (~labeled_all["is_noise"]) & (labeled_all["cluster_id"] >= 0)


def _compute_vdjdb_metrics_for_frame(assignments: pd.DataFrame, metadata_row) -> dict[str, object] | None:
    if assignments.empty:
        return None
    run_id = str(assignments["run_id"].iloc[0])
    epitope = assignments["epitope"].iloc[0]
    cluster_df = assignments.loc[~assignments["is_noise"]]
    labeled_all = assignments.loc[assignments["truth_label"].isin(["positive", "negative"])]
    positives = labeled_all["truth_label"] == "positive"
    negatives = labeled_all["truth_label"] == "negative"
    recovered = _compute_vdjdb_recovered_mask(labeled_all)
    tp = int((positives & recovered).sum())
    fp = int((negatives & recovered).sum())
    fn = int((positives & ~recovered).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    total_labeled = 0
    weighted_purity_sum = 0.0
    positive_counts = []
    positive_clusters = 0
    for _, cluster in cluster_df.groupby("cluster_id"):
        labeled_cluster = cluster.loc[cluster["truth_label"].isin(["positive", "negative"])]
        if labeled_cluster.empty:
            continue
        label_counts = labeled_cluster["truth_label"].value_counts()
        size = int(len(labeled_cluster))
        purity = float(label_counts.max() / size)
        total_labeled += size
        weighted_purity_sum += purity * size
        pos_count = int((labeled_cluster["truth_label"] == "positive").sum())
        if pos_count > 0:
            positive_counts.append(pos_count)
            positive_clusters += 1
    positive_counts = sorted(positive_counts, reverse=True)
    total_positive = max(1, int(positives.sum()))
    return {
        "run_id": run_id,
        "epitope": epitope,
        "method": assignments["method"].iloc[0],
        "parameter_json": assignments["parameter_json"].iloc[0],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weighted_cluster_purity": float(weighted_purity_sum / max(1, total_labeled)),
        "positive_cluster_concentration_top1": float(sum(positive_counts[:1]) / total_positive),
        "positive_cluster_concentration_top5": float(sum(positive_counts[:5]) / total_positive),
        "positive_fragmentation": int(positive_clusters),
        "positive_fragmentation_norm": float(positive_clusters / max(1, tp)),
        "n_clusters": int(metadata_row.get("n_clusters", cluster_df["cluster_id"].nunique())),
        "noise_fraction": float(metadata_row.get("noise_fraction", assignments["is_noise"].mean())),
        "runtime_seconds": float(metadata_row.get("runtime_seconds", np.nan)),
    }


def compute_vdjdb_metrics(assignments: pd.DataFrame, run_metadata: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    rows = []
    metadata_lookup = run_metadata.set_index("run_id", drop=False) if len(run_metadata) else pd.DataFrame()
    for (run_id, epitope), frame in assignments.groupby(["run_id", "epitope"]):
        metadata_row = metadata_lookup.loc[run_id] if len(metadata_lookup) and run_id in metadata_lookup.index else {}
        row = _compute_vdjdb_metrics_for_frame(frame, metadata_row)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def compute_yfv_cluster_enrichment(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    rows = []
    for (run_id, donor_id), frame in assignments.groupby(["run_id", "donor_id"]):
        cluster_df = frame.loc[(~frame["is_noise"]) & (frame["cluster_id"] >= 0)].copy()
        if cluster_df.empty:
            continue
        total_sample = int((frame["sample_label"] == "sample").sum())
        total_background = int((frame["sample_label"] == "background").sum())
        counts = cluster_df.groupby(["cluster_id", "sample_label"]).size().unstack(fill_value=0).reset_index()
        pvals = []
        for _, row in counts.iterrows():
            n_sample = int(row.get("sample", 0))
            n_background = int(row.get("background", 0))
            p_cluster = float(n_background / total_background) if total_background else 0.0
            if p_cluster <= 0.0:
                pval = 0.0 if n_sample > 0 else 1.0
            else:
                pval = float(binom.sf(n_sample - 1, total_sample, p_cluster))
            pvals.append(pval)
        qvals = _fdr_bh(pvals)
        for row, pval, qval in zip(counts.itertuples(index=False), pvals, qvals):
            n_sample = int(getattr(row, "sample", 0))
            n_background = int(getattr(row, "background", 0))
            sample_fraction = float(n_sample / total_sample) if total_sample else 0.0
            background_fraction = float(n_background / total_background) if total_background else 0.0
            log2fc = float(np.log2((sample_fraction + 1e-9) / (background_fraction + 1e-9)))
            rows.append(
                {
                    "run_id": run_id,
                    "donor_id": donor_id,
                    "method": frame["method"].iloc[0],
                    "parameter_json": frame["parameter_json"].iloc[0],
                    "cluster_id": int(row.cluster_id),
                    "n_sample": n_sample,
                    "n_background": n_background,
                    "N_sample": total_sample,
                    "N_background": total_background,
                    "log2FC": log2fc,
                    "p_value": float(pval),
                    "q_value": float(qval),
                    "is_significant": bool(qval < 0.05 and log2fc > 0),
                    "cluster_size": int(n_sample + n_background),
                    "sample_fraction": sample_fraction,
                    "background_fraction": background_fraction,
                }
            )
    return pd.DataFrame(rows)


def summarize_yfv_enrichment(enrichment_df: pd.DataFrame) -> pd.DataFrame:
    if enrichment_df.empty:
        return pd.DataFrame()
    rows = []
    for (run_id, donor_id), frame in enrichment_df.groupby(["run_id", "donor_id"]):
        significant = frame.loc[frame["is_significant"]].copy()
        total_sample = int(frame["N_sample"].iloc[0]) if len(frame) else 0
        retained = int(significant["n_sample"].sum()) if len(significant) else 0
        total_sig = int((significant["n_sample"] + significant["n_background"]).sum()) if len(significant) else 0
        rows.append(
            {
                "run_id": run_id,
                "donor_id": donor_id,
                "method": frame["method"].iloc[0],
                "parameter_json": frame["parameter_json"].iloc[0],
                "number_of_proposed_clusters": int(frame["cluster_id"].nunique()),
                "number_of_significant_enriched_clusters": int(len(significant)),
                "fraction_of_significant_clusters": float(len(significant) / max(1, frame["cluster_id"].nunique())),
                "number_of_retained_post_vaccination_clonotypes": retained,
                "retained_fraction": float(retained / max(1, total_sample)),
                "median_log2FC_enriched": float(significant["log2FC"].median()) if len(significant) else 0.0,
                "max_log2FC_enriched": float(significant["log2FC"].max()) if len(significant) else 0.0,
                "median_q_value_enriched": float(significant["q_value"].median()) if len(significant) else 1.0,
                "background_contamination": float(significant["n_background"].sum() / max(1, total_sig)),
                "weighted_enrichment_score": float(np.average(significant["log2FC"], weights=significant["n_sample"].clip(lower=1))) if len(significant) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_known_yfv_recovery(assignments: pd.DataFrame, enrichment_df: pd.DataFrame, known_yfv: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty or known_yfv.empty:
        return pd.DataFrame()
    def match_key(frame, match_type):
        v_gene = frame["v_gene"].fillna("").astype(str) if "v_gene" in frame.columns else pd.Series("", index=frame.index)
        j_gene = frame["j_gene"].fillna("").astype(str) if "j_gene" in frame.columns else pd.Series("", index=frame.index)
        if match_type == "cdr3_only":
            return frame["cdr3"].astype(str)
        return frame["cdr3"].astype(str) + "|" + v_gene + "|" + j_gene

    rows = []
    enrich_lookup = enrichment_df.set_index(["run_id", "donor_id", "cluster_id"], drop=False)
    for match_type in ["cdr3_only", "cdr3_vj"]:
        assignment_keys = assignments.copy()
        assignment_keys["_match_key"] = match_key(assignment_keys, match_type)
        known_keys = known_yfv.copy()
        known_keys["_match_key"] = match_key(known_keys, match_type)
        for (run_id, donor_id), frame in assignment_keys.groupby(["run_id", "donor_id"]):
            donor_enrichment = enrichment_df.loc[(enrichment_df["run_id"] == run_id) & (enrichment_df["donor_id"] == donor_id)].copy()
            if len(donor_enrichment):
                donor_enrichment["rank_q"] = donor_enrichment["q_value"].rank(method="dense", ascending=True)
                donor_enrichment["rank_log2FC"] = donor_enrichment["log2FC"].rank(method="dense", ascending=False)
            for known_idx, known_row in known_keys.iterrows():
                matched = frame.loc[frame["_match_key"] == known_row["_match_key"]].copy()
                post = matched.loc[matched["sample_label"] == "sample"]
                pre = matched.loc[matched["sample_label"] == "background"]
                candidate = post.loc[~post["is_noise"]]
                candidate_cluster_id = int(candidate["cluster_id"].iloc[0]) if len(candidate) else -1
                enrichment_row = enrich_lookup.loc[(run_id, donor_id, candidate_cluster_id)] if candidate_cluster_id >= 0 and (run_id, donor_id, candidate_cluster_id) in enrich_lookup.index else None
                target_row = donor_enrichment.loc[donor_enrichment["cluster_id"] == candidate_cluster_id].iloc[0] if enrichment_row is not None and len(donor_enrichment.loc[donor_enrichment["cluster_id"] == candidate_cluster_id]) else None
                rows.append(
                    {
                        "run_id": run_id,
                        "donor_id": donor_id,
                        "method": frame["method"].iloc[0],
                        "parameter_json": frame["parameter_json"].iloc[0],
                        "known_yfv_clonotype_id": known_row.get("known_yfv_clonotype_id", known_idx),
                        "cdr3": known_row["cdr3"],
                        "match_type": match_type,
                        "is_present_in_post_sample": bool(len(post) > 0),
                        "is_present_in_pre_sample": bool(len(pre) > 0),
                        "candidate_cluster_id": candidate_cluster_id,
                        "is_candidate_clustered": bool(candidate_cluster_id >= 0),
                        "is_in_significant_cluster": bool(enrichment_row["is_significant"]) if enrichment_row is not None else False,
                        "cluster_q_value": None if enrichment_row is None else float(enrichment_row["q_value"]),
                        "cluster_log2FC": None if enrichment_row is None else float(enrichment_row["log2FC"]),
                        "cluster_rank_by_q_value": None if target_row is None else int(target_row["rank_q"]),
                        "cluster_rank_by_log2FC": None if target_row is None else int(target_row["rank_log2FC"]),
                        "cluster_size": None if enrichment_row is None else int(enrichment_row["cluster_size"]),
                        "cluster_n_sample": None if enrichment_row is None else int(enrichment_row["n_sample"]),
                        "cluster_n_background": None if enrichment_row is None else int(enrichment_row["n_background"]),
                        "cluster_background_fraction": None if enrichment_row is None else float(enrichment_row["background_fraction"]),
                    }
                )
    return pd.DataFrame(rows)


def compute_cross_donor_overlap(assignments: pd.DataFrame, enrichment_df: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    def key(frame):
        v_gene = frame["v_gene"].fillna("").astype(str) if "v_gene" in frame.columns else pd.Series("", index=frame.index)
        j_gene = frame["j_gene"].fillna("").astype(str) if "j_gene" in frame.columns else pd.Series("", index=frame.index)
        chain = frame["chain"].fillna("").astype(str) if "chain" in frame.columns else pd.Series("", index=frame.index)
        return frame["cdr3"].astype(str) + "|" + v_gene + "|" + j_gene + "|" + chain

    rows = []
    assignment_keys = assignments.copy()
    assignment_keys["_key"] = key(assignment_keys)
    for (method, parameter_json), method_assignments in assignment_keys.groupby(["method", "parameter_json"]):
        run_keys = {}
        enriched_keys = {}
        for (run_id, donor_id), frame in method_assignments.groupby(["run_id", "donor_id"]):
            post = frame.loc[frame["sample_label"] == "sample"]
            run_keys[(run_id, donor_id)] = set(post["_key"].tolist())
            sig = set(
                enrichment_df.loc[
                    (enrichment_df["run_id"] == run_id)
                    & (enrichment_df["donor_id"] == donor_id)
                    & (enrichment_df["is_significant"]),
                    "cluster_id",
                ].tolist()
            )
            enriched = post.loc[post["cluster_id"].isin(sig) & ~post["is_noise"]]
            enriched_keys[(run_id, donor_id)] = set(enriched["_key"].tolist())
        for (run_a, donor_a), (run_b, donor_b) in combinations(run_keys.keys(), 2):
            raw_overlap = run_keys[(run_a, donor_a)] & run_keys[(run_b, donor_b)]
            raw_fraction = float(len(raw_overlap) / min(len(run_keys[(run_a, donor_a)]), len(run_keys[(run_b, donor_b)]))) if run_keys[(run_a, donor_a)] and run_keys[(run_b, donor_b)] else 0.0
            enr_overlap = enriched_keys[(run_a, donor_a)] & enriched_keys[(run_b, donor_b)]
            enr_fraction = float(len(enr_overlap) / min(len(enriched_keys[(run_a, donor_a)]), len(enriched_keys[(run_b, donor_b)]))) if enriched_keys[(run_a, donor_a)] and enriched_keys[(run_b, donor_b)] else 0.0
            rows.append(
                {
                    "method": method,
                    "parameter_json": parameter_json,
                    "run_id_a": run_a,
                    "run_id_b": run_b,
                    "donor_a": donor_a,
                    "donor_b": donor_b,
                    "raw_overlap_count": int(len(raw_overlap)),
                    "raw_overlap_fraction": raw_fraction,
                    "enriched_overlap_count": int(len(enr_overlap)),
                    "enriched_overlap_fraction": enr_fraction,
                    "overlap_fold_change": float(enr_fraction / raw_fraction) if raw_fraction else 0.0,
                }
            )
    return pd.DataFrame(rows)
