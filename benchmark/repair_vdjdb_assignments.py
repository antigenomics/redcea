from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VDJDB_DATASETS = ("vdjdb_glc", "vdjdb_ylq")
LABEL_COLUMNS = ["truth_label", "label"]
KEY_COLUMNS = ["clonotype_id", "cdr3", "v_gene", "j_gene", "chain"]


def _load_processed_lookup(processed_dir: Path) -> dict[str, pd.DataFrame]:
    lookups: dict[str, pd.DataFrame] = {}
    for dataset in VDJDB_DATASETS:
        path = processed_dir / f"{dataset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Processed VDJdb dataset is missing: {path}")
        frame = pd.read_parquet(path, columns=KEY_COLUMNS + LABEL_COLUMNS)
        if frame["clonotype_id"].duplicated().any():
            raise ValueError(f"Processed VDJdb dataset has duplicate clonotype_id values: {path}")
        lookups[dataset] = frame.copy()
    return lookups


def _repair_one_file(path: Path, lookups: dict[str, pd.DataFrame]) -> dict[str, object]:
    frame = pd.read_parquet(path)
    if frame.empty:
        return {"path": str(path), "dataset": None, "rows": 0, "updated_rows": 0, "match_key": "empty"}
    dataset_values = frame.get("dataset", pd.Series(dtype="object")).dropna().astype(str).unique().tolist()
    if len(dataset_values) != 1:
        raise ValueError(f"Expected exactly one dataset value in {path}, found: {dataset_values}")
    dataset = dataset_values[0]
    if dataset not in lookups:
        return {"path": str(path), "dataset": dataset, "rows": len(frame), "updated_rows": 0, "match_key": "skipped"}

    lookup = lookups[dataset]
    updated = frame.copy()
    match_key = None

    if "clonotype_id" in updated.columns:
        merged = updated.drop(columns=LABEL_COLUMNS, errors="ignore").merge(
            lookup[["clonotype_id"] + LABEL_COLUMNS],
            on="clonotype_id",
            how="left",
            validate="many_to_one",
        )
        if not merged["truth_label"].isna().any():
            updated = merged
            match_key = "clonotype_id"

    if match_key is None:
        merged = updated.drop(columns=LABEL_COLUMNS, errors="ignore").merge(
            lookup[KEY_COLUMNS[1:] + LABEL_COLUMNS],
            on=KEY_COLUMNS[1:],
            how="left",
            validate="many_to_one",
        )
        if merged["truth_label"].isna().any():
            missing = int(merged["truth_label"].isna().sum())
            raise ValueError(f"Could not fully restore VDJdb labels for {path}; unmatched rows={missing}")
        updated = merged
        match_key = "cdr3_vj_chain"

    updated["label"] = updated["label"].fillna(updated["truth_label"])
    updated.to_parquet(path, index=False)
    return {
        "path": str(path),
        "dataset": dataset,
        "rows": int(len(updated)),
        "updated_rows": int(len(updated)),
        "match_key": match_key,
    }


def repair_vdjdb_assignments(
    *,
    processed_dir: str | Path = "data/processed",
    assignments_dir: str | Path = "results/clustering_assignments",
    report_path: str | Path | None = "results/metrics/vdjdb_assignment_label_repair.tsv",
) -> pd.DataFrame:
    processed_root = Path(processed_dir)
    assignments_root = Path(assignments_dir)
    lookups = _load_processed_lookup(processed_root)
    results: list[dict[str, object]] = []
    for path in sorted(assignments_root.glob("vdjdb_*.parquet")):
        results.append(_repair_one_file(path, lookups))
    report = pd.DataFrame(results)
    if report_path is not None:
        report_out = Path(report_path)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_out, sep="\t", index=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh VDJdb truth labels inside existing benchmark assignment parquet files without rerunning clustering."
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--assignments-dir", default="results/clustering_assignments")
    parser.add_argument("--report-path", default="results/metrics/vdjdb_assignment_label_repair.tsv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = repair_vdjdb_assignments(
        processed_dir=args.processed_dir,
        assignments_dir=args.assignments_dir,
        report_path=args.report_path,
    )
    print(
        "Repaired VDJdb assignment files: files={0}, rows={1}".format(
            len(report),
            int(report["updated_rows"].sum()) if len(report) else 0,
        ),
        flush=True,
    )
    if len(report):
        print(report.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
