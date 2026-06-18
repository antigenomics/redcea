from __future__ import annotations

"""Run the numbered YFV benchmark steps in order via benchmark.yfv_benchmark wrappers."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.yfv_benchmark import (
    plot_yfv_benchmark,
    prepare_yfv_inputs,
    run_yfv_redcea_grid,
    summarize_yfv_benchmark,
    compute_yfv_overlap,
    update_config_summaries,
    write_yfv_benchmark_summary,
    annotate_yfv_llw,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full YFV RedCEA benchmark pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nproc", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_yfv_inputs(args.config, args.outdir, dry_run=args.dry_run)
    run_yfv_redcea_grid(args.config, args.outdir, dry_run=args.dry_run, nproc=args.nproc)
    if args.dry_run:
        return 0
    annotate_yfv_llw(args.config, args.outdir)
    summarize_yfv_benchmark(args.config, args.outdir)
    compute_yfv_overlap(args.config, args.outdir)
    update_config_summaries(args.outdir)
    plot_yfv_benchmark(args.config, args.outdir)
    write_yfv_benchmark_summary(args.config, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
