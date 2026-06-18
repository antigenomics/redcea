from __future__ import annotations

"""Thin wrapper around benchmark.yfv_benchmark.summarize_yfv_benchmark and config rollups."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.yfv_benchmark import summarize_yfv_benchmark, update_config_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write donor- and LLW-level YFV benchmark summaries.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summarize_yfv_benchmark(args.config, args.outdir)
    update_config_summaries(args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
