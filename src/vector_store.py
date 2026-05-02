from __future__ import annotations

import chromadb
from . import config

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return _client


def get_collection(repo_slug: str) -> chromadb.Collection:
    return _get_client().get_or_create_collection(repo_slug)


def upsert(
    repo_slug: str,
    ids: list[str],
    docs: list[str],
    vecs: list[list[float]],
    metas: list[dict],
) -> None:
    get_collection(repo_slug).upsert(
        ids=ids, documents=docs, embeddings=vecs, metadatas=metas
    )


def query(
    repo_slug: str,
    vector: list[float],
    k: int,
    where: dict | None = None,
) -> dict:
    col = get_collection(repo_slug)
    n = min(k, col.count())
    if n == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    kwargs: dict = {
        "query_embeddings": [vector],
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    return col.query(**kwargs)


def dump_all(repo_slug: str) -> dict:
    """Return all stored ids + documents for BM25 rebuild."""
    return get_collection(repo_slug).get(include=["documents"])


def count(repo_slug: str) -> int:
    return get_collection(repo_slug).count()


def list_repos() -> list[tuple[str, int]]:
    """Returns [(repo_slug, issue_count), ...] for all indexed repos."""
    client = _get_client()
    out = []
    for col in client.list_collections():
        name = col.name if hasattr(col, "name") else str(col)
        out.append((name, client.get_collection(name).count()))
    return out


def clear(repo_slug: str | None = None) -> None:
    client = _get_client()
    if repo_slug:
        try:
            client.delete_collection(repo_slug)
        except Exception:
            pass
    else:
        for col in client.list_collections():
            name = col.name if hasattr(col, "name") else str(col)
            try:
                client.delete_collection(name)
            except Exception:
                pass