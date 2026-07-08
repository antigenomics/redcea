from __future__ import annotations

"""Thin wrapper around benchmark.yfv_benchmark.prepare_yfv_inputs."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark.yfv_benchmark import prepare_yfv_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized YFV inputs and a dataset manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_yfv_inputs(args.config, args.outdir, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
