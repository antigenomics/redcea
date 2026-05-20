from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.data_sources import (
    DEFAULT_TCRVDB_PADJ_THRESHOLD,
    DEFAULT_TCRVDB_PATH,
    DEFAULT_VDJDB_BG_SOURCE_AIRR,
    DEFAULT_VDJDB_BG_SOURCE_EMBEDDING,
    DEFAULT_VDJDB_BG_VJ_AIRR,
    DEFAULT_VDJDB_BG_VJ_EMBEDDING,
    DEFAULT_VDJDB_EMBED_DIR,
    DEFAULT_VDJDB_FULL_PATH,
    DEFAULT_VDJDB_RELEASE_PATH,
    DEFAULT_YFV_AIRR_DIR,
    DEFAULT_YFV_KNOWN_EPITOPES,
    DEFAULT_YFV_RUNS_DIR,
    VDJDB_TARGETS,
    discover_yfv_donor_ids,
    resolve_vdjdb_background_paths,
    resolve_vdjdb_embedding_path,
    resolve_tcrvdb_path,
    resolve_yfv_embedding_paths,
)


def normalize_segment(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
        .str.replace("/", "_", regex=False)
    )


def first_present(frame: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def infer_chain(frame: pd.DataFrame, default: str = "TRB") -> pd.Series:
    if "chain" in frame.columns:
        return frame["chain"].fillna(default).astype(str)
    if "locus" in frame.columns:
        locus = frame["locus"].fillna(default).astype(str)
        return locus.replace({"beta": "TRB", "alpha": "TRA"})
    return pd.Series([default] * len(frame), index=frame.index, dtype="object")


def is_metadata_numeric_column(column: object) -> bool:
    name = str(column).lower()
    metadata_tokens = [
        "id",
        "index",
        "clone",
        "count",
        "freq",
        "fraction",
        "size",
        "length",
        "len",
        "timepoint",
        "padj",
        "pvalue",
        "qvalue",
    ]
    return any(token in name for token in metadata_tokens)


def infer_numeric_embedding_columns(frame: pd.DataFrame) -> list[object]:
    numeric_columns = list(frame.select_dtypes(include=[np.number]).columns)
    return [column for column in numeric_columns if not is_metadata_numeric_column(column)]


def split_embedding_metadata(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    emb_cols = [column for column in frame.columns if str(column).startswith("emb_")]
    if "embedding" in frame.columns:
        embedding_array = np.asarray(frame["embedding"].tolist(), dtype=np.float32)
        metadata = frame.drop(columns=["embedding"]).copy()
        return metadata, embedding_array
    if emb_cols:
        embedding_array = frame[emb_cols].to_numpy(dtype=np.float32, copy=False)
        metadata = frame.drop(columns=emb_cols).copy()
        return metadata, embedding_array
    numeric_embedding_cols = infer_numeric_embedding_columns(frame)
    if numeric_embedding_cols:
        embedding_array = frame[numeric_embedding_cols].to_numpy(dtype=np.float32, copy=False)
        metadata = frame.drop(columns=numeric_embedding_cols).copy()
        return metadata, embedding_array
    raise KeyError(
        "Expected an 'embedding' column, 'emb_*' columns, or numeric coordinate columns in embedding parquet."
    )


def add_embedding_columns(frame: pd.DataFrame, embedding_array: np.ndarray) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    for index in range(embedding_array.shape[1]):
        out[f"emb_{index}"] = embedding_array[:, index]
    return out


def read_airr_like_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".gz":
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def standardize_metadata_frame(frame: pd.DataFrame, *, chain_default: str = "TRB") -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["cdr3"] = first_present(frame, ["cdr3", "cdr3aa", "junction_aa", "cdr3_beta_aa", "cdr3_alpha_aa"])
    out["v_gene"] = normalize_segment(first_present(frame, ["v_gene", "v_call", "v.segm", "TRBV", "TRBV_IMGT", "v"]))
    out["j_gene"] = normalize_segment(first_present(frame, ["j_gene", "j_call", "j.segm", "TRBJ", "TRBJ_IMGT", "j"]))
    out["chain"] = infer_chain(frame, default=chain_default)
    if "clone_id" in frame.columns:
        out["source_clone_id"] = frame["clone_id"].astype(str)
    elif "clonotype_id" in frame.columns:
        out["source_clone_id"] = frame["clonotype_id"].astype(str)
    else:
        out["source_clone_id"] = pd.Series(frame.index.astype(str), index=frame.index, dtype="object")
    return out


def clone_id_candidates(frame: pd.DataFrame, *, sample_label: Optional[str] = None) -> pd.DataFrame:
    candidates = pd.DataFrame(index=frame.index)
    raw_ids = None
    if "clone_id" in frame.columns:
        raw_ids = frame["clone_id"].astype(str)
        candidates["clone_id"] = raw_ids
    elif "clonotype_id" in frame.columns:
        raw_ids = frame["clonotype_id"].astype(str)
        candidates["clone_id"] = raw_ids

    one_based = pd.Series(np.arange(1, len(frame) + 1), index=frame.index).astype(str)
    zero_based = pd.Series(np.arange(len(frame)), index=frame.index).astype(str)
    candidates["row_1_based"] = one_based
    candidates["row_0_based"] = zero_based
    if sample_label is not None:
        prefix = "s_" if sample_label == "sample" else "b_"
        candidates[f"{prefix}row_1_based"] = prefix + one_based
        candidates[f"{prefix}row_0_based"] = prefix + zero_based
        if raw_ids is not None:
            candidates[f"{prefix}clone_id"] = prefix + raw_ids.str.replace(r"^[sb]_", "", regex=True)
    return candidates


def best_clone_id_key(left: pd.DataFrame, right: pd.DataFrame) -> tuple[Optional[tuple[str, str]], int]:
    best_column = None
    best_overlap = 0
    for left_column in left.columns:
        left_values = set(left[left_column].dropna().astype(str))
        if not left_values:
            continue
        for right_column in right.columns:
            overlap = len(left_values & set(right[right_column].dropna().astype(str)))
            if overlap > best_overlap:
                best_column = (left_column, right_column)
                best_overlap = overlap
    return best_column, best_overlap


def build_vdjdb_truth_table(tcrvdb_path: str | Path, *, padj_threshold: float = DEFAULT_TCRVDB_PADJ_THRESHOLD) -> pd.DataFrame:
    truth = pd.read_csv(resolve_tcrvdb_path(tcrvdb_path)).drop(columns=["Unnamed: 0"], errors="ignore")
    truth["chain"] = "TRB"
    truth["cdr3"] = truth["cdr3_beta_aa"].fillna("").astype(str)
    truth["v_gene"] = normalize_segment(truth["TRBV"])
    truth["j_gene"] = normalize_segment(truth["TRBJ"])
    truth["truth_label"] = np.where(
        truth["padj"].notna(),
        np.where(truth["padj"] < float(padj_threshold), "positive", "negative"),
        "unlabeled",
    )
    return truth[
        [
            "cdr3",
            "v_gene",
            "j_gene",
            "chain",
            "epitope_aa",
            "truth_label",
            "padj",
            "data_origin",
            "name",
        ]
    ].copy()


def assemble_vdjdb_processed_dataset(
    target_key: str,
    *,
    vdjdb_embed_dir: str | Path = DEFAULT_VDJDB_EMBED_DIR,
    tcrvdb_path: str | Path = DEFAULT_TCRVDB_PATH,
    padj_threshold: float = DEFAULT_TCRVDB_PADJ_THRESHOLD,
) -> pd.DataFrame:
    resolved = resolve_vdjdb_embedding_path(target_key, vdjdb_embed_dir)
    epitope_sequence = resolved["epitope_sequence"]
    embedding_frame = pd.read_parquet(resolved["sample_embedding"])
    metadata_from_embedding, embedding_array = split_embedding_metadata(embedding_frame)
    standardized = standardize_metadata_frame(metadata_from_embedding, chain_default="TRB")
    truth = build_vdjdb_truth_table(tcrvdb_path, padj_threshold=padj_threshold)
    truth_ep = truth.loc[(truth["epitope_aa"] == epitope_sequence) & (truth["chain"] == "TRB")].copy()
    truth_ep = truth_ep.drop_duplicates(subset=["cdr3", "v_gene", "j_gene", "chain"], keep="first")
    merged = standardized.merge(
        truth_ep[["cdr3", "v_gene", "j_gene", "chain", "truth_label"]],
        on=["cdr3", "v_gene", "j_gene", "chain"],
        how="left",
    )
    merged["truth_label"] = merged["truth_label"].fillna("unlabeled")
    merged["epitope"] = epitope_sequence
    merged["label"] = merged["truth_label"]
    merged["embedding_id"] = np.arange(len(merged), dtype=np.int64)
    merged["clonotype_id"] = [f"{target_key.lower()}_{i:06d}" for i in range(len(merged))]
    merged["cdr3_length"] = merged["cdr3"].fillna("").astype(str).str.len()
    merged = add_embedding_columns(merged, embedding_array)
    return merged[
        [
            "clonotype_id",
            "cdr3",
            "v_gene",
            "j_gene",
            "chain",
            "epitope",
            "label",
            "truth_label",
            "embedding_id",
            "cdr3_length",
            *[column for column in merged.columns if column.startswith("emb_")],
        ]
    ].copy()


def align_yfv_embedding_with_airr(
    donor_id: str,
    *,
    sample_label: str,
    embedding_path: Path,
    airr_path: Path,
) -> pd.DataFrame:
    embedding_frame = pd.read_parquet(embedding_path)
    metadata_from_embedding, embedding_array = split_embedding_metadata(embedding_frame)
    airr_frame = read_airr_like_table(airr_path).reset_index(drop=True)
    standardized_airr = standardize_metadata_frame(airr_frame, chain_default="TRB")
    standardized_embedding = standardize_metadata_frame(metadata_from_embedding, chain_default="TRB")
    airr_candidates = clone_id_candidates(airr_frame, sample_label=sample_label)
    embedding_candidates = clone_id_candidates(metadata_from_embedding, sample_label=sample_label)
    match_columns, overlap = best_clone_id_key(embedding_candidates, airr_candidates)
    if match_columns is not None and overlap == len(metadata_from_embedding):
        embedding_key, airr_key = match_columns
        airr_with_key = standardized_airr.copy()
        airr_with_key["_merge_clone_id"] = airr_candidates[airr_key].astype(str)
        embedding_order = pd.DataFrame(
            {
                "_merge_clone_id": embedding_candidates[embedding_key].astype(str),
                "_embedding_order": np.arange(len(metadata_from_embedding), dtype=np.int64),
            }
        )
        merged = embedding_order.merge(airr_with_key, on="_merge_clone_id", how="left", sort=False)
        if merged[["cdr3", "v_gene", "j_gene"]].isna().any(axis=None):
            raise ValueError(
                "Failed to align AIRR metadata by clone_id for donor {0} {1}: matched={2}, embedding_rows={3}.".format(
                    donor_id,
                    sample_label,
                    overlap,
                    len(metadata_from_embedding),
                )
            )
        merged = merged.sort_values("_embedding_order").drop(columns=["_merge_clone_id", "_embedding_order"]).reset_index(drop=True)
    elif len(airr_frame) == len(embedding_frame):
        merged = standardized_airr.copy()
    else:
        raise ValueError(
            "Cannot align AIRR metadata for donor {0} {1}: embedding rows={2}, AIRR rows={3}, best clone_id overlap={4}.".format(
                donor_id,
                sample_label,
                len(embedding_frame),
                len(airr_frame),
                overlap,
            )
        )
    for column in ["cdr3", "v_gene", "j_gene", "chain"]:
        empty_mask = merged[column].fillna("").astype(str).eq("")
        merged.loc[empty_mask, column] = standardized_embedding.loc[empty_mask, column]
    subject, replicate = donor_id.split("_", 1)
    merged["donor_id"] = donor_id
    merged["timepoint"] = "15" if sample_label == "sample" else "0"
    merged["sample_type"] = "post" if sample_label == "sample" else "pre"
    merged["sample_label"] = sample_label
    merged["embedding_id"] = np.arange(len(merged), dtype=np.int64)
    merged["clonotype_id"] = [f"{donor_id.lower()}_{sample_label}_{i:06d}" for i in range(len(merged))]
    merged["cdr3_length"] = merged["cdr3"].fillna("").astype(str).str.len()
    merged["subject_id"] = subject
    merged["replicate_id"] = replicate
    merged = add_embedding_columns(merged, embedding_array)
    return merged[
        [
            "clonotype_id",
            "donor_id",
            "subject_id",
            "replicate_id",
            "timepoint",
            "sample_type",
            "sample_label",
            "cdr3",
            "v_gene",
            "j_gene",
            "chain",
            "embedding_id",
            "cdr3_length",
            *[column for column in merged.columns if column.startswith("emb_")],
        ]
    ].copy()


def assemble_yfv_processed_dataset(
    *,
    yfv_runs_dir: str | Path = DEFAULT_YFV_RUNS_DIR,
    yfv_airr_dir: str | Path = DEFAULT_YFV_AIRR_DIR,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    donor_ids = discover_yfv_donor_ids(yfv_runs_dir)
    for donor_id in donor_ids:
        resolved = resolve_yfv_embedding_paths(donor_id, yfv_runs_dir)
        subject, replicate = donor_id.split("_", 1)
        sample_airr = Path(yfv_airr_dir) / f"{subject}_15_{replicate}.txt"
        background_airr = Path(yfv_airr_dir) / f"{subject}_0_{replicate}_with_1.txt"
        rows.append(
            align_yfv_embedding_with_airr(
                donor_id,
                sample_label="sample",
                embedding_path=resolved["sample_embedding"],
                airr_path=sample_airr,
            )
        )
        rows.append(
            align_yfv_embedding_with_airr(
                donor_id,
                sample_label="background",
                embedding_path=resolved["background_embedding"],
                airr_path=background_airr,
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_known_yfv_clonotypes(
    tcrvdb_path: str | Path,
    *,
    known_epitopes: tuple[str, ...] = DEFAULT_YFV_KNOWN_EPITOPES,
) -> pd.DataFrame:
    truth = pd.read_csv(resolve_tcrvdb_path(tcrvdb_path)).drop(columns=["Unnamed: 0"], errors="ignore")
    truth = truth.loc[(truth["data_origin"] == "vdjdb") & truth["epitope_aa"].isin(known_epitopes)].copy()
    truth["cdr3"] = truth["cdr3_beta_aa"].fillna("").astype(str)
    truth["v_gene"] = normalize_segment(truth["TRBV_IMGT"].fillna(truth["TRBV"]))
    truth["j_gene"] = normalize_segment(truth["TRBJ_IMGT"].fillna(truth["TRBJ"]))
    grouped = (
        truth.groupby(["epitope_aa", "cdr3", "v_gene", "j_gene"])
        .size()
        .reset_index(name="support")
        .sort_values(["epitope_aa", "support", "cdr3"], ascending=[True, False, True])
    )
    top = grouped.groupby("epitope_aa", as_index=False).head(1).reset_index(drop=True)
    top["known_yfv_clonotype_id"] = [f"known_yfv_{row.epitope_aa.lower()}_top1" for row in top.itertuples()]
    top["source"] = "VDJdb"
    return top[
        [
            "known_yfv_clonotype_id",
            "cdr3",
            "v_gene",
            "j_gene",
            "epitope_aa",
            "source",
            "support",
        ]
    ].rename(columns={"epitope_aa": "epitope"})


def build_source_manifest(
    yfv_runs_dir: str | Path = DEFAULT_YFV_RUNS_DIR,
    yfv_airr_dir: str | Path = DEFAULT_YFV_AIRR_DIR,
    vdjdb_embed_dir: str | Path = DEFAULT_VDJDB_EMBED_DIR,
    tcrvdb_path: str | Path = DEFAULT_TCRVDB_PATH,
    vdjdb_release_path: str | Path = DEFAULT_VDJDB_RELEASE_PATH,
    vdjdb_full_path: str | Path = DEFAULT_VDJDB_FULL_PATH,
    vdjdb_bg_source_airr: str | Path = DEFAULT_VDJDB_BG_SOURCE_AIRR,
    vdjdb_bg_source_embedding: str | Path = DEFAULT_VDJDB_BG_SOURCE_EMBEDDING,
    vdjdb_bg_vj_airr: str | Path = DEFAULT_VDJDB_BG_VJ_AIRR,
    vdjdb_bg_vj_embedding: str | Path = DEFAULT_VDJDB_BG_VJ_EMBEDDING,
) -> pd.DataFrame:
    rows = []
    background_paths = resolve_vdjdb_background_paths(
        source_airr=vdjdb_bg_source_airr,
        source_embedding=vdjdb_bg_source_embedding,
        vj_airr=vdjdb_bg_vj_airr,
        vj_embedding=vdjdb_bg_vj_embedding,
    )
    for source_type, path, required in [
        ("truth_labels", resolve_tcrvdb_path(tcrvdb_path), True),
        ("vdjdb_release", Path(vdjdb_release_path), True),
        ("vdjdb_full", Path(vdjdb_full_path), True),
        ("background_source_airr", background_paths["source_airr"], True),
        ("background_source_embedding", background_paths["source_embedding"], True),
        ("background_vj_airr", background_paths["vj_airr"], False),
        ("background_vj_embedding", background_paths["vj_embedding"], False),
    ]:
        rows.append(
            {
                "dataset_group": "vdjdb",
                "dataset_id": "TRB",
                "sample_label": "background" if "background" in source_type else "reference",
                "source_type": source_type,
                "path": str(path),
                "exists": path.exists(),
                "required": bool(required),
            }
        )

    for target_key in sorted(VDJDB_TARGETS):
        resolved = resolve_vdjdb_embedding_path(target_key, vdjdb_embed_dir)
        rows.extend(
            [
                {
                    "dataset_group": "vdjdb",
                    "dataset_id": target_key,
                    "sample_label": "sample",
                    "source_type": "embedding",
                    "path": str(resolved["sample_embedding"]),
                    "exists": resolved["sample_embedding"].exists(),
                    "required": True,
                },
                {
                    "dataset_group": "vdjdb",
                    "dataset_id": target_key,
                    "sample_label": "sample",
                    "source_type": "index",
                    "path": str(resolved["sample_index"]),
                    "exists": resolved["sample_index"].exists(),
                    "required": True,
                },
            ]
        )

    donor_ids = discover_yfv_donor_ids(yfv_runs_dir)
    for donor_id in donor_ids:
        resolved = resolve_yfv_embedding_paths(donor_id, yfv_runs_dir)
        for sample_label, key in [
            ("sample", "sample_embedding"),
            ("background", "background_embedding"),
            ("sample", "sample_index"),
            ("background", "background_index"),
        ]:
            rows.append(
                {
                    "dataset_group": "yfv",
                    "dataset_id": donor_id,
                    "sample_label": sample_label,
                    "source_type": "embedding" if "embedding" in key else "index",
                    "path": str(resolved[key]),
                    "exists": Path(resolved[key]).exists(),
                    "required": True,
                }
            )
        subject, replicate = donor_id.split("_", 1)
        sample_airr = Path(yfv_airr_dir) / f"{subject}_15_{replicate}.txt"
        background_airr = Path(yfv_airr_dir) / f"{subject}_0_{replicate}_with_1.txt"
        rows.extend(
            [
                {
                    "dataset_group": "yfv",
                    "dataset_id": donor_id,
                    "sample_label": "sample",
                    "source_type": "airr",
                    "path": str(sample_airr),
                    "exists": sample_airr.exists(),
                    "required": True,
                },
                {
                    "dataset_group": "yfv",
                    "dataset_id": donor_id,
                    "sample_label": "background",
                    "source_type": "airr",
                    "path": str(background_airr),
                    "exists": background_airr.exists(),
                    "required": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def write_processed_datasets(
    *,
    processed_dir: str | Path = "data/processed",
    yfv_runs_dir: str | Path = DEFAULT_YFV_RUNS_DIR,
    yfv_airr_dir: str | Path = DEFAULT_YFV_AIRR_DIR,
    vdjdb_embed_dir: str | Path = DEFAULT_VDJDB_EMBED_DIR,
    tcrvdb_path: str | Path = DEFAULT_TCRVDB_PATH,
    padj_threshold: float = DEFAULT_TCRVDB_PADJ_THRESHOLD,
) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    glc = assemble_vdjdb_processed_dataset(
        "GLC",
        vdjdb_embed_dir=vdjdb_embed_dir,
        tcrvdb_path=tcrvdb_path,
        padj_threshold=padj_threshold,
    )
    ylq = assemble_vdjdb_processed_dataset(
        "YLQ",
        vdjdb_embed_dir=vdjdb_embed_dir,
        tcrvdb_path=tcrvdb_path,
        padj_threshold=padj_threshold,
    )
    yfv = assemble_yfv_processed_dataset(yfv_runs_dir=yfv_runs_dir, yfv_airr_dir=yfv_airr_dir)
    known_yfv = build_known_yfv_clonotypes(tcrvdb_path)

    output_paths = {
        "vdjdb_glc": processed_dir / "vdjdb_glc.parquet",
        "vdjdb_ylq": processed_dir / "vdjdb_ylq.parquet",
        "yfv_repertoires": processed_dir / "yfv_repertoires.parquet",
        "known_yfv": processed_dir / "known_yfv_vdjdb_clonotypes.tsv",
    }
    glc.to_parquet(output_paths["vdjdb_glc"], index=False)
    ylq.to_parquet(output_paths["vdjdb_ylq"], index=False)
    yfv.to_parquet(output_paths["yfv_repertoires"], index=False)
    known_yfv.to_csv(output_paths["known_yfv"], sep="\t", index=False)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and assemble benchmark source datasets.")
    parser.add_argument("--yfv-runs-dir", default=str(DEFAULT_YFV_RUNS_DIR))
    parser.add_argument("--yfv-airr-dir", default=str(DEFAULT_YFV_AIRR_DIR))
    parser.add_argument("--vdjdb-embed-dir", default=str(DEFAULT_VDJDB_EMBED_DIR))
    parser.add_argument("--tcrvdb-path", default=str(DEFAULT_TCRVDB_PATH))
    parser.add_argument("--vdjdb-release-path", default=str(DEFAULT_VDJDB_RELEASE_PATH))
    parser.add_argument("--vdjdb-full-path", default=str(DEFAULT_VDJDB_FULL_PATH))
    parser.add_argument("--vdjdb-bg-source-airr", default=str(DEFAULT_VDJDB_BG_SOURCE_AIRR))
    parser.add_argument("--vdjdb-bg-source-embedding", default=str(DEFAULT_VDJDB_BG_SOURCE_EMBEDDING))
    parser.add_argument("--vdjdb-bg-vj-airr", default=str(DEFAULT_VDJDB_BG_VJ_AIRR))
    parser.add_argument("--vdjdb-bg-vj-embedding", default=str(DEFAULT_VDJDB_BG_VJ_EMBEDDING))
    parser.add_argument("--padj-threshold", type=float, default=DEFAULT_TCRVDB_PADJ_THRESHOLD)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--manifest-out", default="results/run_metadata/source_parquet_manifest.tsv")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any expected source path is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_source_manifest(
        yfv_runs_dir=args.yfv_runs_dir,
        yfv_airr_dir=args.yfv_airr_dir,
        vdjdb_embed_dir=args.vdjdb_embed_dir,
        tcrvdb_path=args.tcrvdb_path,
        vdjdb_release_path=args.vdjdb_release_path,
        vdjdb_full_path=args.vdjdb_full_path,
        vdjdb_bg_source_airr=args.vdjdb_bg_source_airr,
        vdjdb_bg_source_embedding=args.vdjdb_bg_source_embedding,
        vdjdb_bg_vj_airr=args.vdjdb_bg_vj_airr,
        vdjdb_bg_vj_embedding=args.vdjdb_bg_vj_embedding,
    )
    out_path = Path(args.manifest_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, sep="\t", index=False)
    if "required" not in manifest.columns:
        manifest["required"] = True
    missing = manifest.loc[manifest["required"] & ~manifest["exists"]].copy()
    if len(missing):
        print("Missing expected source files:")
        print(missing.to_string(index=False))
        if args.strict:
            raise SystemExit(1)

    output_paths = write_processed_datasets(
        processed_dir=args.processed_dir,
        yfv_runs_dir=args.yfv_runs_dir,
        yfv_airr_dir=args.yfv_airr_dir,
        vdjdb_embed_dir=args.vdjdb_embed_dir,
        tcrvdb_path=args.tcrvdb_path,
        padj_threshold=args.padj_threshold,
    )
    for name, path in output_paths.items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
