from __future__ import annotations

import json
from pathlib import Path
from . import config


def _path(repo_slug: str) -> Path:
    config.STATE_PATH.mkdir(parents=True, exist_ok=True)
    return config.STATE_PATH / f"{repo_slug}.json"


def load(repo_slug: str) -> dict | None:
    p = _path(repo_slug)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save(repo_slug: str, last_updated: str, count: int) -> None:
    _path(repo_slug).write_text(
        json.dumps({"last_updated": last_updated, "issue_count": count}, indent=2)
    )
