from __future__ import annotations

"""Thin wrapper around benchmark.yfv_benchmark.write_yfv_benchmark_summary."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.yfv_benchmark import write_yfv_benchmark_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the unified YFV benchmark markdown summary.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_yfv_benchmark_summary(args.config, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
