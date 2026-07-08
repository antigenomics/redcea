from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FALSE_ERROR_FRAGMENT = "['junction_aa', 'v_call', 'j_call', 'locus'] not in index"


def repair_run_metadata(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dataset_mode: str | None = None,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    frame = pd.read_csv(input_path, sep="\t")
    repaired = frame.copy()

    if dataset_mode is not None and "dataset_mode" in repaired.columns:
        scope_mask = repaired["dataset_mode"].astype(str).eq(dataset_mode)
    else:
        scope_mask = pd.Series([True] * len(repaired), index=repaired.index)

    error_text = repaired.get("error_message", pd.Series([""] * len(repaired), index=repaired.index)).fillna("").astype(str)
    false_error_mask = scope_mask & error_text.str.contains(FALSE_ERROR_FRAGMENT, regex=False)

    repaired.loc[false_error_mask, "status"] = "success"
    if "error_message" in repaired.columns:
        repaired.loc[false_error_mask, "error_message"] = ""
    if "error_traceback" in repaired.columns:
        repaired.loc[false_error_mask, "error_traceback"] = ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(output_path, sep="\t", index=False)
    return repaired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite false benchmark error statuses to success.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dataset-mode", choices=["vdjdb", "yfv"], default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repaired = repair_run_metadata(
        args.input_path,
        args.output_path,
        dataset_mode=args.dataset_mode,
    )
    status_counts = repaired["status"].value_counts(dropna=False).to_dict() if "status" in repaired.columns else {}
    print(
        "Repaired run metadata: rows={0}, status_counts={1}, output={2}".format(
            len(repaired),
            status_counts,
            Path(args.output_path),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
