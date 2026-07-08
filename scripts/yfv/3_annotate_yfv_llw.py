from __future__ import annotations

"""Thin wrapper around benchmark.yfv_benchmark.annotate_yfv_llw."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.yfv_benchmark import annotate_yfv_llw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate YFV runs with LLW/VDJdb exact-match hits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotate_yfv_llw(args.config, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
