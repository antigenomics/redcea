from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def execute_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    env: dict[str, object] = {"__name__": "__main__"}
    execution_count = 1
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        cell["execution_count"] = execution_count
        execution_count += 1
        if not source.strip():
            cell["outputs"] = []
            continue
        try:
            exec(compile(source, str(path), "exec"), env, env)
            cell["outputs"] = []
        except Exception:
            exc = sys.exc_info()[1]
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                }
            ]
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            raise
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a notebook locally without nbconvert.")
    parser.add_argument("notebook_path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute_notebook(Path(args.notebook_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
