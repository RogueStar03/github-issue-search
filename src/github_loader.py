from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from github import Github, Auth

load_dotenv()


def slug(repo: str) -> str:
    """Convert 'owner/name' to a filesystem- and Chroma-safe 'owner__name'."""
    return repo.replace("/", "__")


def fetch_issues(
    repo: str,
    since: datetime | None = None,
    limit: int | None = None,
) -> list[dict]:
    """
    Fetch issues from a public GitHub repo.
    Returns list of dicts (no comments in v1 — saves API budget).
    """
    token = os.environ.get("GITHUB_TOKEN")
    gh = Github(auth=Auth.Token(token)) if token else Github()

    r = gh.get_repo(repo)
    kwargs: dict = {"state": "all", "sort": "updated", "direction": "asc"}
    if since:
        kwargs["since"] = since

    issues_iter = r.get_issues(**kwargs)
    results = []
    for issue in issues_iter:
        if issue.pull_request is not None:
            continue
        results.append({
            "number": issue.number,
            "title": issue.title,
            "body": (issue.body or ""),
            "state": issue.state,
            "labels": ",".join(l.name for l in issue.labels),
            "created_at": issue.created_at.isoformat(),
            "updated_at": issue.updated_at.isoformat(),
            "url": issue.html_url,
        })
        if limit and len(results) >= limit:
            break

    return results