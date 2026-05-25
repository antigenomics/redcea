from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.airr_utils import normalize_segment, read_airr_like_table, standardize_metadata_frame, to_tcremp_airr_frame
from benchmark.data_sources import (
    DEFAULT_TCRVDB_PADJ_THRESHOLD,
    DEFAULT_TCRVDB_PATH,
    DEFAULT_VDJDB_AIRR_DIR,
    DEFAULT_VDJDB_BG_SOURCE_AIRR,
    DEFAULT_VDJDB_BG_SOURCE_EMBEDDING,
    DEFAULT_VDJDB_EMBED_DIR,
    DEFAULT_YFV_KNOWN_EPITOPES,
    DEFAULT_YFV_RUNS_DIR,
    VDJDB_TARGETS,
    discover_yfv_donor_ids,
    resolve_tcrvdb_path,
    resolve_vdjdb_embedding_path,
    resolve_vdjdb_rep_path,
    resolve_yfv_embedding_paths,
)


def log_step(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[{0}] {1}".format(timestamp, message), flush=True)


def build_vdjdb_truth_table(
    tcrvdb_path: str | Path,
    *,
    padj_threshold: float = DEFAULT_TCRVDB_PADJ_THRESHOLD,
) -> pd.DataFrame:
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


def validate_tcremp_airr_columns(airr_path: Path) -> None:
    frame = read_airr_like_table(airr_path)
    standardized = to_tcremp_airr_frame(frame, chain_default="TRB")
    required_columns = ["junction_aa", "v_call", "j_call", "locus"]
    for column in required_columns:
        normalized = standardized[column].fillna("").astype(str).str.strip()
        if normalized.eq("").any():
            raise ValueError(
                "Expected non-empty tcremp column {0} in {1}, but found empty rows={2}".format(
                    column,
                    airr_path,
                    int(normalized.eq("").sum()),
                )
            )
    standardize_metadata_frame(standardized, chain_default="TRB")


def parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required to read parquet row counts efficiently") from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


def materialize_standardized_airr_table(
    source_path: Path,
    output_path: Path,
    *,
    chain_default: str = "TRB",
) -> Path:
    if output_path.exists():
        try:
            if output_path.stat().st_mtime >= source_path.stat().st_mtime:
                log_step("Reusing normalized AIRR table: {0}".format(output_path))
                return output_path
        except FileNotFoundError:
            pass
    log_step("Normalizing AIRR table: {0} -> {1}".format(source_path, output_path))
    frame = read_airr_like_table(source_path)
    standardized = to_tcremp_airr_frame(frame, chain_default=chain_default)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(output_path, sep="\t", index=False)
    log_step("Wrote normalized AIRR rows={0}: {1}".format(len(standardized), output_path))
    return output_path


def validate_embedding_airr_pair(airr_path: Path, embedding_path: Path) -> int:
    airr_path = Path(airr_path)
    embedding_path = Path(embedding_path)
    if not airr_path.exists():
        raise FileNotFoundError("AIRR file is missing: {0}".format(airr_path))
    if not embedding_path.exists():
        raise FileNotFoundError("Embedding parquet is missing: {0}".format(embedding_path))
    log_step("Validating AIRR/parquet pair: airr={0}, embedding={1}".format(airr_path, embedding_path))
    validate_tcremp_airr_columns(airr_path)
    airr_rows = len(read_airr_like_table(airr_path))
    embedding_rows = parquet_row_count(embedding_path)
    if airr_rows != embedding_rows:
        raise ValueError(
            "AIRR/embedding row mismatch: airr_rows={0}, embedding_rows={1}, airr_path={2}, embedding_path={3}".format(
                airr_rows,
                embedding_rows,
                airr_path,
                embedding_path,
            )
        )
    log_step("Validated row counts: rows={0}".format(airr_rows))
    return embedding_rows


def build_dataset_manifest(
    *,
    yfv_runs_dir: str | Path = DEFAULT_YFV_RUNS_DIR,
    vdjdb_embed_dir: str | Path = DEFAULT_VDJDB_EMBED_DIR,
    vdjdb_airr_dir: str | Path = DEFAULT_VDJDB_AIRR_DIR,
    vdjdb_bg_airr: str | Path = DEFAULT_VDJDB_BG_SOURCE_AIRR,
    vdjdb_bg_embedding: str | Path = DEFAULT_VDJDB_BG_SOURCE_EMBEDDING,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    vdjdb_background_airr = Path(vdjdb_bg_airr)
    vdjdb_background_embedding = Path(vdjdb_bg_embedding)
    log_step("Validating shared VDJdb background inputs")
    validate_embedding_airr_pair(vdjdb_background_airr, vdjdb_background_embedding)

    for target_key in sorted(VDJDB_TARGETS):
        log_step("Preparing VDJdb dataset target={0}".format(target_key))
        resolved = resolve_vdjdb_embedding_path(target_key, vdjdb_embed_dir)
        rep_path = resolve_vdjdb_rep_path(target_key, vdjdb_airr_dir)
        validate_embedding_airr_pair(rep_path, resolved["sample_embedding"])
        rows.append(
            {
                "dataset": f"vdjdb_{target_key.lower()}",
                "dataset_mode": "vdjdb",
                "dataset_key": target_key,
                "epitope": resolved["epitope_sequence"],
                "donor_id": None,
                "chain": "TRB",
                "species": "HomoSapiens",
                "sample_airr_path": str(rep_path),
                "background_airr_path": str(vdjdb_background_airr),
                "sample_embedding_path": str(resolved["sample_embedding"]),
                "background_embedding_path": str(vdjdb_background_embedding),
                "sample_index_path": str(resolved["sample_index"]),
                "background_index_path": str(vdjdb_background_embedding.with_suffix(".index")),
                "background_kind": "vdjdb_motifs_trb_background_100k",
            }
        )

    donor_ids = discover_yfv_donor_ids(yfv_runs_dir)
    log_step("Discovered YFV donors count={0} in {1}".format(len(donor_ids), yfv_runs_dir))
    for donor_id in donor_ids:
        log_step("Preparing YFV dataset donor={0}".format(donor_id))
        resolved = resolve_yfv_embedding_paths(donor_id, yfv_runs_dir)
        sample_airr = Path(resolved["sample_representation"])
        background_airr = Path(resolved["background_representation"])
        validate_embedding_airr_pair(sample_airr, resolved["sample_embedding"])
        validate_embedding_airr_pair(background_airr, resolved["background_embedding"])
        rows.append(
            {
                "dataset": "yfv_repertoires",
                "dataset_mode": "yfv",
                "dataset_key": donor_id,
                "epitope": None,
                "donor_id": donor_id,
                "chain": "TRB",
                "species": "HomoSapiens",
                "sample_airr_path": str(sample_airr),
                "background_airr_path": str(background_airr),
                "sample_embedding_path": str(resolved["sample_embedding"]),
                "background_embedding_path": str(resolved["background_embedding"]),
                "sample_index_path": str(resolved["sample_index"]),
                "background_index_path": str(resolved["background_index"]),
                "background_kind": "paired_pre_vaccination_repertoire",
            }
        )

    return pd.DataFrame(rows)


def write_processed_datasets(
    *,
    processed_dir: str | Path = "data/processed",
    yfv_runs_dir: str | Path = DEFAULT_YFV_RUNS_DIR,
    vdjdb_embed_dir: str | Path = DEFAULT_VDJDB_EMBED_DIR,
    vdjdb_airr_dir: str | Path = DEFAULT_VDJDB_AIRR_DIR,
    vdjdb_bg_airr: str | Path = DEFAULT_VDJDB_BG_SOURCE_AIRR,
    vdjdb_bg_embedding: str | Path = DEFAULT_VDJDB_BG_SOURCE_EMBEDDING,
    tcrvdb_path: str | Path = DEFAULT_TCRVDB_PATH,
) -> dict[str, Path]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    normalized_yfv_airr_dir = processed_dir / "yfv_airr"
    normalized_yfv_airr_dir.mkdir(parents=True, exist_ok=True)

    yfv_runs_dir = Path(yfv_runs_dir)
    donor_ids = discover_yfv_donor_ids(yfv_runs_dir)
    log_step("Starting dataset preparation; processed_dir={0}".format(processed_dir))
    log_step("Discovered YFV donors for normalization count={0} in {1}".format(len(donor_ids), yfv_runs_dir))
    for donor_id in donor_ids:
        log_step("Materializing normalized YFV AIRR donor={0}".format(donor_id))
        resolved = resolve_yfv_embedding_paths(donor_id, yfv_runs_dir)
        materialize_standardized_airr_table(
            Path(resolved["sample_representation"]),
            normalized_yfv_airr_dir / "{0}_sample.tsv".format(donor_id),
        )
        materialize_standardized_airr_table(
            Path(resolved["background_representation"]),
            normalized_yfv_airr_dir / "{0}_background.tsv".format(donor_id),
        )

    dataset_manifest = build_dataset_manifest(
        yfv_runs_dir=yfv_runs_dir,
        vdjdb_embed_dir=vdjdb_embed_dir,
        vdjdb_airr_dir=vdjdb_airr_dir,
        vdjdb_bg_airr=vdjdb_bg_airr,
        vdjdb_bg_embedding=vdjdb_bg_embedding,
    )
    if len(dataset_manifest):
        yfv_mask = dataset_manifest["dataset_mode"].eq("yfv")
        dataset_manifest.loc[yfv_mask, "sample_airr_path"] = dataset_manifest.loc[yfv_mask, "donor_id"].map(
            lambda donor_id: str(normalized_yfv_airr_dir / "{0}_sample.tsv".format(donor_id))
        )
        dataset_manifest.loc[yfv_mask, "background_airr_path"] = dataset_manifest.loc[yfv_mask, "donor_id"].map(
            lambda donor_id: str(normalized_yfv_airr_dir / "{0}_background.tsv".format(donor_id))
        )
    known_yfv = build_known_yfv_clonotypes(tcrvdb_path)
    output_paths = {
        "dataset_manifest": processed_dir / "benchmark_dataset_manifest.tsv",
        "known_yfv": processed_dir / "known_yfv_vdjdb_clonotypes.tsv",
    }
    dataset_manifest.to_csv(output_paths["dataset_manifest"], sep="\t", index=False)
    known_yfv.to_csv(output_paths["known_yfv"], sep="\t", index=False)
    log_step("Wrote dataset manifest rows={0}: {1}".format(len(dataset_manifest), output_paths["dataset_manifest"]))
    log_step("Wrote known YFV clonotypes rows={0}: {1}".format(len(known_yfv), output_paths["known_yfv"]))
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the benchmark dataset manifest.")
    parser.add_argument("--yfv-runs-dir", default=str(DEFAULT_YFV_RUNS_DIR))
    parser.add_argument("--vdjdb-embed-dir", default=str(DEFAULT_VDJDB_EMBED_DIR))
    parser.add_argument("--vdjdb-airr-dir", default=str(DEFAULT_VDJDB_AIRR_DIR))
    parser.add_argument("--vdjdb-bg-airr", default=str(DEFAULT_VDJDB_BG_SOURCE_AIRR))
    parser.add_argument("--vdjdb-bg-embedding", default=str(DEFAULT_VDJDB_BG_SOURCE_EMBEDDING))
    parser.add_argument("--tcrvdb-path", default=str(DEFAULT_TCRVDB_PATH))
    parser.add_argument("--processed-dir", default="data/processed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_paths = write_processed_datasets(
        processed_dir=args.processed_dir,
        yfv_runs_dir=args.yfv_runs_dir,
        vdjdb_embed_dir=args.vdjdb_embed_dir,
        vdjdb_airr_dir=args.vdjdb_airr_dir,
        vdjdb_bg_airr=args.vdjdb_bg_airr,
        vdjdb_bg_embedding=args.vdjdb_bg_embedding,
        tcrvdb_path=args.tcrvdb_path,
    )
    for name, path in output_paths.items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
