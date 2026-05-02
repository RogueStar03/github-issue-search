from __future__ import annotations

import json
import bm25s
from . import config

_loaded: dict[str, bm25s.BM25] = {}


def _index_dir(repo_slug: str):
    d = config.BM25_PATH / repo_slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(repo_slug: str, ids: list[str], docs: list[str]) -> None:
    """Build BM25 index from all docs and save to disk. Always rebuilds fully."""
    if not docs:
        return
    corpus_tokens = bm25s.tokenize(docs, stopwords="en")
    r = bm25s.BM25()
    r.index(corpus_tokens)
    # Save index only (no corpus) so retrieve() returns integer indices
    r.save(str(_index_dir(repo_slug)))
    _save_ids(repo_slug, ids)
    _loaded[repo_slug] = r


def _load(repo_slug: str) -> bm25s.BM25 | None:
    d = _index_dir(repo_slug)
    if not d.exists() or not any(d.iterdir()):
        return None
    r = bm25s.BM25.load(str(d), mmap=True)
    _loaded[repo_slug] = r
    return r


def query(repo_slug: str, q: str, k: int) -> list[tuple[str, float]]:
    """Returns [(doc_id, score), ...] sorted by score descending."""
    r = _loaded.get(repo_slug) or _load(repo_slug)
    if r is None:
        return []

    ids = _load_ids(repo_slug)
    if not ids:
        return []

    q_tokens = bm25s.tokenize([q], stopwords="en")
    actual_k = min(k, len(ids))
    # retrieve without corpus → results are integer indices into original corpus
    results, scores = r.retrieve(q_tokens, k=actual_k)

    out = []
    for idx, score in zip(results[0], scores[0]):
        idx = int(idx)
        if 0 <= idx < len(ids):
            out.append((ids[idx], float(score)))
    return out


def save_ids(repo_slug: str, ids: list[str]) -> None:
    _save_ids(repo_slug, ids)


def _save_ids(repo_slug: str, ids: list[str]) -> None:
    (_index_dir(repo_slug) / "ids.json").write_text(json.dumps(ids))


def _load_ids(repo_slug: str) -> list[str]:
    p = _index_dir(repo_slug) / "ids.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())