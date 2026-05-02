from __future__ import annotations

from datetime import datetime, timezone

from . import github_loader, embeddings, vector_store, bm25_index, state, config


def _build_doc(issue: dict) -> str:
    """Combine title + body into a single embeddable string, truncated."""
    text = f"{issue['title']}\n\n{issue['body']}"
    return text[: config.MAX_BODY_CHARS]


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def index_repo(repo: str, limit: int | None = None) -> dict:
    """
    Fetch issues from GitHub, embed them, upsert into ChromaDB, and rebuild BM25.
    Returns {"new": int, "total": int}.
    """
    slug = github_loader.slug(repo)
    cp = state.load(slug)
    since = _parse_iso(cp["last_updated"]) if cp else None

    issues = github_loader.fetch_issues(repo, since=since, limit=limit)
    if not issues:
        return {"new": 0, "total": vector_store.count(slug)}

    docs = [_build_doc(i) for i in issues]
    vecs = embeddings.embed(docs)
    metas = [
        {
            "title": i["title"],
            "state": i["state"],
            "labels": i["labels"],
            "created_at": i["created_at"],
            "updated_at": i["updated_at"],
            "url": i["url"],
        }
        for i in issues
    ]
    ids = [str(i["number"]) for i in issues]

    vector_store.upsert(slug, ids, docs, vecs, metas)

    # Rebuild BM25 from entire corpus (incremental updates not supported by bm25s)
    all_data = vector_store.dump_all(slug)
    bm25_index.build(slug, all_data["ids"], all_data["documents"])

    last_updated = max(i["updated_at"] for i in issues)
    total = vector_store.count(slug)
    state.save(slug, last_updated=last_updated, count=total)

    return {"new": len(issues), "total": total}