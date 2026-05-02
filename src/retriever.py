from __future__ import annotations

from . import embeddings, vector_store, bm25_index, config


def _rrf(ranked_lists: list[list[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(
    repo_slug: str,
    query: str,
    k_final: int = config.TOP_K_FINAL,
) -> list[dict]:
    """
    Hybrid search: vector cosine + BM25, fused via RRF.
    Returns list of result dicts with issue metadata and rank info.
    """
    qvec = embeddings.embed([query])[0]

    # Vector retrieval
    vec_result = vector_store.query(repo_slug, qvec, k=config.TOP_K_VECTOR)
    vec_ids = vec_result["ids"][0] if vec_result["ids"] else []
    vec_metas = vec_result["metadatas"][0] if vec_result["metadatas"] else []
    vec_docs = vec_result["documents"][0] if vec_result["documents"] else []

    # BM25 retrieval
    bm25_hits = bm25_index.query(repo_slug, query, k=config.TOP_K_BM25)
    bm25_ids = [doc_id for doc_id, _ in bm25_hits]

    # RRF fusion
    fused = _rrf([vec_ids, bm25_ids])

    # Build rank lookup for debug output
    vec_rank_map = {doc_id: i for i, doc_id in enumerate(vec_ids)}
    bm25_rank_map = {doc_id: i for i, doc_id in enumerate(bm25_ids)}

    # Build id → metadata map from vector results (already have them)
    meta_map: dict[str, dict] = {}
    doc_map: dict[str, str] = {}
    for doc_id, meta, doc in zip(vec_ids, vec_metas, vec_docs):
        meta_map[doc_id] = meta
        doc_map[doc_id] = doc

    # For ids only in BM25, fetch from Chroma
    bm25_only = set(bm25_ids) - set(vec_ids)
    if bm25_only:
        col = vector_store.get_collection(repo_slug)
        extra = col.get(ids=list(bm25_only), include=["documents", "metadatas"])
        for doc_id, meta, doc in zip(extra["ids"], extra["metadatas"], extra["documents"]):
            meta_map[doc_id] = meta
            doc_map[doc_id] = doc

    results = []
    for doc_id, rrf_score in fused[:k_final]:
        meta = meta_map.get(doc_id, {})
        results.append({
            "number": int(doc_id),
            "title": meta.get("title", ""),
            "body_excerpt": (doc_map.get(doc_id, ""))[:300],
            "state": meta.get("state", ""),
            "labels": meta.get("labels", ""),
            "url": meta.get("url", ""),
            "rrf_score": round(rrf_score, 4),
            "vector_rank": vec_rank_map.get(doc_id, -1),
            "bm25_rank": bm25_rank_map.get(doc_id, -1),
        })

    return results