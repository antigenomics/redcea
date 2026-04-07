from pathlib import Path
import sys


def _ensure_local_tcremp_on_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sibling_repo = repo_root.parent / "tcremp"
    if sibling_repo.exists():
        sibling_path = str(sibling_repo)
        if sibling_path not in sys.path:
            sys.path.insert(0, sibling_path)


_ensure_local_tcremp_on_path()

from tcremp import get_resource_path

__all__ = ["get_resource_path"]
