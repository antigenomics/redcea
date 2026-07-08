from __future__ import annotations

"""Thin wrapper around benchmark.yfv_benchmark.compute_yfv_overlap."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.yfv_benchmark import compute_yfv_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute raw-vs-RedCEA YFV cross-donor overlap.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compute_yfv_overlap(args.config, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
