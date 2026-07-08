from __future__ import annotations

"""Thin wrapper around benchmark.run_benchmark execution via benchmark.yfv_benchmark.run_yfv_redcea_grid."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.yfv_benchmark import run_yfv_redcea_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the YFV RedCEA benchmark grid.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nproc", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_yfv_redcea_grid(args.config, args.outdir, dry_run=args.dry_run, nproc=args.nproc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
