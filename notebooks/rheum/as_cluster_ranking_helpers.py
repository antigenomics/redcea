from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
from sklearn.preprocessing import MinMaxScaler

try:
    import logomaker
except Exception:  # pragma: no cover - optional notebook dependency
    logomaker = None

try:
    from mir.common.clonotype import ClonotypeAA
    from mir.common.clonotype_dataset import ClonotypeDataset
except Exception:  # pragma: no cover - optional dependency in notebooks
    ClonotypeAA = None
    ClonotypeDataset = None


DEFAULT_AS_SEQS = [
    "CASSVGLFSTDTQYF",
    "CASSVGLYSTDTQYF",
    "CASSAGLFSTDTQYF",
    "CASSAGLYSTDTQYF",
    "CASSLGLFSTDTQYF",
    "CASSLGLYSTDTQYF",
    "CASSPGLFSTDTQYF",
    "CASSPGLYSTDTQYF",
]


def _pgen_worker(seq: str) -> float:
    try:
        from mir.basic.pgen import OlgaModel

        model = OlgaModel()
        return float(model.compute_pgen_cdr3aa(seq))
    except Exception:
        return np.nan


def compute_pgen_pool(
    df: pd.DataFrame,
    *,
    seq_col: str = "cdr3aa_beta",
    out_col: str = "pgen",
    processes: int = 32,
    chunksize: int = 500,
) -> pd.Series:
    from multiprocessing import Pool

    sequences = df[seq_col].astype(str).tolist()
    with Pool(processes=processes) as pool:
        values = list(pool.imap(_pgen_worker, sequences, chunksize=chunksize))
    return pd.Series(values, index=df.index, name=out_col)


def compute_topsis_score(
    df: pd.DataFrame,
    metric_types: dict[str, str],
    *,
    weight_dict: dict[str, float] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    metrics = list(metric_types.keys())

    normalized = df[metrics].copy()
    scaler = MinMaxScaler()
    normalized[metrics] = scaler.fit_transform(normalized[metrics])

    for metric, kind in metric_types.items():
        if kind == "cost":
            normalized[metric] = 1 - normalized[metric]

    if weight_dict is None:
        weights = np.ones(len(metrics)) / len(metrics)
    else:
        weights = np.array([weight_dict[metric] for metric in metrics], dtype=float)
        weights = weights / weights.sum()

    weighted = normalized * weights
    ideal = weighted.max()
    anti_ideal = weighted.min()
    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - anti_ideal, axis=1)
    df["topsis_score"] = d_neg / (d_pos + d_neg)
    return df.sort_values("topsis_score", ascending=False).reset_index(drop=True)


def plot_volcano(
    df: pd.DataFrame,
    *,
    pval_threshold: float = 0.05,
    fold_threshold: float = 0.0,
    sample_name: str = "",
    ax=None,
    layers: list[dict] | None = None,
    point_size: int | None = None,
) -> plt.Axes:
    df = df.copy()
    eps = 1e-10
    df["log10_pval"] = -np.log10(df["enrichment_pvalue_zbinom"] + eps)
    df["significant"] = (
        (df["enrichment_fdr_zbinom"] < pval_threshold)
        & (df["log_fold_change"] > fold_threshold)
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="log_fold_change",
        y="log10_pval",
        hue="significant",
        palette={True: "#ffcb9a", False: "#d2e8e3"},
        edgecolor="black",
        linewidth=0.3,
        ax=ax,
        **({"s": point_size} if point_size else {}),
    )

    if layers:
        for layer in layers:
            column = layer["col"]
            mask = df[column].fillna(False).astype(bool)
            if not mask.any():
                continue
            sns.scatterplot(
                data=df.loc[mask],
                x="log_fold_change",
                y="log10_pval",
                color=layer.get("color", "#c1121f"),
                marker=layer.get("marker", "o"),
                label=layer.get("label", column),
                edgecolor="black",
                linewidth=0.4,
                ax=ax,
                **({"s": point_size} if point_size else {}),
            )

    ax.axvline(fold_threshold, linestyle="--", color="black")
    ax.axhline(-np.log10(pval_threshold + eps), linestyle="--", color="black")
    ax.set_title(sample_name)
    ax.set_xlabel("log2(Fold Enrichment)")
    ax.set_ylabel("-log10(p-value)")
    return ax


def plot_logo(clonotypes: pd.Series) -> None:
    if logomaker is None:
        raise ImportError("logomaker is required for sequence logo plots")
    matrix = logomaker.alignment_to_matrix(clonotypes.dropna().astype(str))
    logomaker.Logo(matrix, color_scheme="skylign_protein", ax=plt.gca())


@dataclass
class RunTables:
    sample_name: str
    summary: pd.DataFrame
    clusters: pd.DataFrame
    enriched: pd.DataFrame


def normalize_gene(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\*.*$", "", str(value))


def infer_source_from_sample_name(sample_name: str, case_regex: str, control_regex: str) -> str:
    if re.search(case_regex, sample_name):
        return "case"
    if re.search(control_regex, sample_name):
        return "control"
    return "other"


def load_run_tables(run_dir: str | Path) -> RunTables:
    run_dir = Path(run_dir)
    prefix = run_dir.name

    summary = pd.read_csv(run_dir / f"{prefix}_summary_tcrempnet.tsv", sep="\t")
    clusters = pd.read_csv(run_dir / f"{prefix}_tcremp_clusters.tsv", sep="\t")

    enriched_candidates = [
        run_dir / f"{prefix}_enriched_clonotypes_tcremp_pgen.tsv",
        run_dir / f"{prefix}_enriched_clonotypes_tcremp.tsv",
    ]
    enriched_path = next((path for path in enriched_candidates if path.exists()), None)
    if enriched_path is None:
        raise FileNotFoundError(f"No enriched clonotype file found in {run_dir}")

    enriched = pd.read_csv(enriched_path, sep="\t")
    return RunTables(sample_name=prefix, summary=summary, clusters=clusters, enriched=enriched)


def load_many_runs(runs_root: str | Path, sample_names: list[str] | None = None) -> dict[str, RunTables]:
    runs_root = Path(runs_root)
    if sample_names is None:
        sample_names = sorted(path.name for path in runs_root.iterdir() if path.is_dir())
    return {sample_name: load_run_tables(runs_root / sample_name) for sample_name in sample_names}


def build_reference_matcher(reference_sequences: list[str]):
    if ClonotypeAA is not None and ClonotypeDataset is not None:
        ref_clonotypes = [ClonotypeAA(cdr3aa=seq) for seq in reference_sequences]
        return ClonotypeDataset(ref_clonotypes)
    return set(reference_sequences)


def has_reference_match(sequence: str, matcher, threshold: int = 1) -> bool:
    if sequence is None or pd.isna(sequence):
        return False

    sequence = str(sequence)
    if hasattr(matcher, "get_matching_clonotypes"):
        return len(matcher.get_matching_clonotypes(sequence, threshold=threshold)) > 0

    reference_sequences: set[str] = matcher
    if threshold <= 0:
        return sequence in reference_sequences

    for ref in reference_sequences:
        if len(ref) != len(sequence):
            continue
        mismatches = sum(a != b for a, b in zip(ref, sequence))
        if mismatches <= threshold:
            return True
    return False


def annotate_as_pattern(
    df: pd.DataFrame,
    matcher,
    *,
    seq_col: str = "cdr3aa_beta",
    threshold: int = 1,
    out_col: str = "as_pattern",
) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = df[seq_col].apply(lambda value: has_reference_match(value, matcher, threshold=threshold))
    return df


def annotate_run_volcano_summary(run: RunTables, matcher, *, as_match_threshold: int = 1) -> pd.DataFrame:
    clusters = annotate_as_pattern(run.clusters, matcher, threshold=as_match_threshold)
    as_by_cluster = (
        clusters.groupby("cluster_id")["as_pattern"]
        .any()
        .rename("as_pattern")
        .reset_index()
    )
    summary = run.summary.merge(as_by_cluster, on="cluster_id", how="left")
    summary["as_pattern"] = summary["as_pattern"].fillna(False)
    summary["sample_name"] = run.sample_name
    return summary


def plot_sample_volcano_grid(
    runs: dict[str, RunTables],
    matcher,
    *,
    samples_per_row: int = 3,
    fold_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    as_match_threshold: int = 1,
    figsize_per_panel: tuple[float, float] = (6.0, 4.5),
) -> tuple[plt.Figure, list[pd.DataFrame]]:
    sample_names = list(runs)
    n_samples = len(sample_names)
    n_cols = samples_per_row
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    summaries = []
    for idx, sample_name in enumerate(sample_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        summary = annotate_run_volcano_summary(
            runs[sample_name],
            matcher,
            as_match_threshold=as_match_threshold,
        )
        plot_volcano(
            summary,
            pval_threshold=pval_threshold,
            fold_threshold=fold_threshold,
            sample_name=sample_name,
            ax=ax,
            layers=[
                {
                    "col": "as_pattern",
                    "label": f"AS clone <= {as_match_threshold} aa mismatch",
                    "color": "#c1121f",
                    "marker": "o",
                }
            ],
            point_size=45,
        )
        summaries.append(summary)

    for idx in range(n_samples, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.tight_layout()
    return fig, summaries


def _wildcard_neighbors(sequence: str, substitutions: int) -> set[str]:
    if substitutions <= 0:
        return {sequence}

    keys = {sequence}
    for idxs in combinations(range(len(sequence)), substitutions):
        chars = list(sequence)
        for idx in idxs:
            chars[idx] = "*"
        keys.add("".join(chars))
    return keys


def merge_clusters_by_cdr3_threshold(
    df: pd.DataFrame,
    *,
    substitutions: int = 0,
    cluster_col: str = "cluster_uid",
    seq_col: str = "cdr3aa_beta",
    merged_col: str = "merged_cluster_id",
) -> tuple[pd.DataFrame, dict[str, int]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    seq_key_to_clusters: dict[tuple[int, str], set[str]] = defaultdict(set)

    for _, row in df[[cluster_col, seq_col]].drop_duplicates().iterrows():
        cluster_id = row[cluster_col]
        sequence = str(row[seq_col])
        wildcard_keys = {sequence} if substitutions <= 0 else _wildcard_neighbors(sequence, substitutions)
        for wildcard_key in wildcard_keys:
            seq_key_to_clusters[(len(sequence), wildcard_key)].add(cluster_id)

    all_clusters = set(df[cluster_col].astype(str))
    for clusters in seq_key_to_clusters.values():
        clusters = sorted(str(cluster) for cluster in clusters)
        if len(clusters) < 2:
            continue
        first = clusters[0]
        for other in clusters[1:]:
            adjacency[first].add(other)
            adjacency[other].add(first)

    visited: set[str] = set()
    mapping: dict[str, int] = {}
    next_id = 0

    for cluster_id in sorted(all_clusters):
        if cluster_id in visited:
            continue
        stack = [cluster_id]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)

        for member in component:
            mapping[member] = next_id
        next_id += 1

    merged_df = df.copy()
    merged_df[cluster_col] = merged_df[cluster_col].astype(str)
    merged_df[merged_col] = merged_df[cluster_col].map(mapping)
    return merged_df, mapping


def prepare_merged_clonotypes(
    runs: dict[str, RunTables],
    matcher,
    *,
    compute_pgen_if_missing: bool = False,
    pgen_processes: int = 16,
    as_match_threshold: int = 1,
    case_regex: str = r"^as_",
    control_regex: str = r"^(?:h_|hd_)",
) -> pd.DataFrame:
    frames = []
    for sample_name, run in runs.items():
        sample_enriched = run.enriched.copy()
        sample_enriched["sample_name"] = sample_name
        sample_enriched["sample_group"] = infer_source_from_sample_name(
            sample_name,
            case_regex=case_regex,
            control_regex=control_regex,
        )
        sample_enriched["cluster_uid"] = (
            sample_enriched["sample_name"].astype(str)
            + "::"
            + sample_enriched["cluster_id"].astype(str)
        )
        sample_enriched["as_pattern"] = sample_enriched["cdr3aa_beta"].apply(
            lambda value: has_reference_match(value, matcher, threshold=as_match_threshold)
        )
        sample_enriched["v_gene"] = sample_enriched["v_beta"].map(normalize_gene)
        sample_enriched["j_gene"] = sample_enriched["j_beta"].map(normalize_gene)
        frames.append(sample_enriched)

    df = pd.concat(frames, ignore_index=True)
    if "pgen" not in df.columns:
        df["pgen"] = np.nan

    if compute_pgen_if_missing and df["pgen"].isna().all():
        df["pgen"] = compute_pgen_pool(df, processes=pgen_processes)

    return df


def _safe_ratio(numerator: float, denominator: float, pseudo: float = 0.5) -> float:
    return (numerator + pseudo) / (denominator + pseudo)


def summarize_merged_clusters(
    df: pd.DataFrame,
    *,
    merged_col: str = "merged_cluster_id",
    source_col: str = "source",
    case_group: str = "case",
    control_group: str = "control",
) -> pd.DataFrame:
    cluster_sizes = df.groupby(merged_col).size().rename("cluster_size")
    sample_counts = (
        df.groupby(merged_col)
        .agg(
            n_samples=("sample_name", "nunique"),
            as_pattern=("as_pattern", "any"),
            pgen=("pgen", "mean"),
        )
    )
    source_counts = (
        df.pivot_table(
            index=merged_col,
            columns=source_col,
            values="clone_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename_axis(columns=None)
    )

    summary = (
        cluster_sizes.to_frame()
        .join(sample_counts, how="left")
        .join(source_counts, how="left")
        .reset_index()
    )

    for column in ("sample", "background"):
        if column not in summary.columns:
            summary[column] = 0

    summary["sample_fraction"] = summary["sample"] / summary["cluster_size"].clip(lower=1)
    summary["background_fraction"] = summary["background"] / summary["cluster_size"].clip(lower=1)

    group_presence = (
        df.groupby([merged_col, "sample_group", "sample_name"])
        .size()
        .reset_index(name="n")
    )

    case_usage = (
        group_presence[group_presence["sample_group"] == case_group]
        .groupby(merged_col)["sample_name"]
        .nunique()
        .rename("as_usage")
    )
    control_usage = (
        group_presence[group_presence["sample_group"] == control_group]
        .groupby(merged_col)["sample_name"]
        .nunique()
        .rename("h_usage")
    )

    summary = summary.join(case_usage, on=merged_col).join(control_usage, on=merged_col)
    summary["as_usage"] = summary["as_usage"].fillna(0).astype(int)
    summary["h_usage"] = summary["h_usage"].fillna(0).astype(int)

    total_case = max(df.loc[df["sample_group"] == case_group, "sample_name"].nunique(), 1)
    total_control = max(df.loc[df["sample_group"] == control_group, "sample_name"].nunique(), 1)

    fisher_p_values = []
    patient_fc_values = []
    for _, row in summary.iterrows():
        as_used = int(row["as_usage"])
        h_used = int(row["h_usage"])
        contingency = [
            [as_used, max(total_case - as_used, 0)],
            [h_used, max(total_control - h_used, 0)],
        ]
        _, p_value = fisher_exact(contingency)
        fisher_p_values.append(p_value)

        case_prev = as_used / total_case
        control_prev = h_used / total_control
        patient_fc_values.append(_safe_ratio(case_prev, control_prev))

    summary["fisher_p"] = fisher_p_values
    summary["patient_fc"] = patient_fc_values
    summary["log_patient_fc"] = np.log10(summary["patient_fc"])
    summary["log_pgen"] = np.log10(summary["pgen"].clip(lower=1e-16))
    summary["log_cluster_size"] = np.log10(summary["cluster_size"].clip(lower=1))

    return summary.sort_values(["n_samples", "cluster_size"], ascending=False).reset_index(drop=True)


def rank_clusters_with_topsis(
    summary: pd.DataFrame,
    *,
    min_samples: int = 2,
    metric_types: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    if metric_types is None:
        metric_types = {
            "sample_fraction": "benefit",
            "as_usage": "benefit",
            "h_usage": "cost",
            "log_pgen": "cost",
            "patient_fc": "benefit",
            "log_cluster_size": "benefit",
        }

    ranked_input = summary.loc[summary["n_samples"] >= min_samples].copy()
    ranked = compute_topsis_score(ranked_input, metric_types, weight_dict=weights)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def plot_ranked_metrics(
    ranked_df: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    hue_col: str = "as_pattern",
    figsize: tuple[float, float] = (14.0, 8.0),
) -> plt.Figure:
    if metrics is None:
        metrics = [
            "topsis_score",
            "patient_fc",
            "sample_fraction",
            "as_usage",
            "h_usage",
            "log_pgen",
            "cluster_size",
        ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    sns.scatterplot(data=ranked_df, x="rank", y="topsis_score", hue=hue_col, ax=axes[0], palette="Set1")
    axes[0].set_title("TOPSIS score by rank")

    sns.scatterplot(data=ranked_df, x="rank", y="patient_fc", hue=hue_col, ax=axes[1], palette="Set1", legend=False)
    axes[1].set_title("Patient fold-change by rank")

    metric_frame = ranked_df[["rank"] + metrics].melt(id_vars="rank", var_name="metric", value_name="value")
    sns.lineplot(data=metric_frame, x="rank", y="value", hue="metric", ax=axes[2])
    axes[2].set_title("Metric trajectories across rank")

    top_slice = ranked_df.head(min(25, len(ranked_df)))
    sns.scatterplot(
        data=top_slice,
        x="sample_fraction",
        y="patient_fc",
        size="cluster_size",
        hue=hue_col,
        palette="Set1",
        ax=axes[3],
    )
    axes[3].set_title("Top clusters: sample fraction vs patient FC")

    fig.tight_layout()
    return fig


def summarize_vj_genes(cluster_df: pd.DataFrame, *, top_k: int = 3) -> str:
    v_counts = Counter(cluster_df["v_gene"].dropna())
    j_counts = Counter(cluster_df["j_gene"].dropna())
    top_v = ", ".join(f"{gene}({count})" for gene, count in v_counts.most_common(top_k)) or "NA"
    top_j = ", ".join(f"{gene}({count})" for gene, count in j_counts.most_common(top_k)) or "NA"
    return f"V: {top_v}\nJ: {top_j}"


def plot_top_cluster_logos(
    df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    *,
    top_n: int = 12,
    merged_col: str = "merged_cluster_id",
    seq_col: str = "cdr3aa_beta",
    figsize_per_panel: tuple[float, float] = (4.5, 3.5),
) -> plt.Figure:
    top_clusters = ranked_df.head(top_n)
    n_cols = 3
    n_rows = int(np.ceil(len(top_clusters) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )

    for idx, row in top_clusters.reset_index(drop=True).iterrows():
        ax = axes[idx // n_cols][idx % n_cols]
        cluster_id = row[merged_col]
        cluster_df = df[df[merged_col] == cluster_id]
        plt.sca(ax)
        plot_logo(cluster_df[seq_col].dropna())
        ax.set_title(
            f"rank {int(row['rank'])} | cluster {cluster_id}\n"
            f"TOPSIS={row['topsis_score']:.3f} | n={len(cluster_df)}\n"
            f"{summarize_vj_genes(cluster_df)}",
            fontsize=10,
        )

    for idx in range(len(top_clusters), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.tight_layout()
    return fig
