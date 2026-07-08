import argparse
import sys
from pathlib import Path
from datetime import datetime
import time

import pandas as pd
from multiprocessing import Pool
import numpy as np

from mir.basic.pgen import OlgaModel
olga = OlgaModel()

def _pgen_worker(seq: str):
    try:
        return float(olga.compute_pgen_cdr3aa(seq))
    except Exception:
        return np.nan


def compute_pgen_pool(df: pd.DataFrame,
                      seq_col: str = "cdr3aa_beta",
                      out_col: str = "pgen",
                      processes: int = 32,
                      chunksize: int = 500) -> pd.Series:
    seqs = df[seq_col].astype(str).tolist()

    with Pool(processes=processes) as pool:
        vals = list(pool.imap(_pgen_worker, seqs, chunksize=chunksize))

    return pd.Series(vals, index=df.index, name=out_col)


def process_file(path: Path, processes: int, inplace: bool) -> Path:
    print(f"Processing {path} ...")
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started at {start_dt}")

    df = pd.read_csv(path, sep="\t").head(1_000_000)

    if "cdr3aa_beta" not in df.columns:
        raise ValueError(f"File {path} does not contain column 'cdr3aa_beta'.")

    df["pgen"] = compute_pgen_pool(
        df,
        seq_col="cdr3aa_beta",
        out_col="pgen",
        processes=processes,
        chunksize=500,
    )

    if inplace:
        out_path = path
    else:
        out_path = path.with_name(path.stem + "_pgen.tsv")

    df.to_csv(out_path, sep="\t", index=False)

    end_time = time.time()
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    print(f"Finished at {end_dt} (elapsed {duration:.2f} seconds)")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Add pgen column to all *_enriched_clonotypes_tcremp.tsv files"
    )
    parser.add_argument(
        "--name", "-n", required=True,
        help="Project name / subdirectory in /projects/immunestatus/{name}/tcrempnet"
    )
    parser.add_argument(
        "--processes", "-p", type=int, default=8,
        help="Number of processes for compute_pgen_pool (default: 8)"
    )
    parser.add_argument(
        "--inplace", action="store_true", default=False,
        help="Overwrite input files instead of creating new *_pgen.tsv files"
    )
    parser.add_argument(
        "--file", '-f', action="store_true", default=False,
        help="Path to one file for processing"
    )
    args = parser.parse_args()
    
    if not args.file:
        base = Path("/projects/immunestatus") / args.name / "tcrempnet"
        pattern = "*_enriched_clonotypes_tcremp.tsv"
        files = sorted(base.glob(pattern))
    else:
        base = None
        pattern = None
        files = [Path(args.name)]

    if not files:
        if args.file:
            print(f"No file found: {args.name}", file=sys.stderr)
        else:
            print(f"No files found matching {base}/{pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} files")

    for f in files:
        try:
            out_f = process_file(f, args.processes, args.inplace)
            print(f"[OK] {f.name} -> {out_f.name}")
        except Exception as e:
            print(f"[FAIL] {f}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
