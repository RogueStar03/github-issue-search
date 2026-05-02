from __future__ import annotations

from . import retriever, llm, config

_SYSTEM = """\
You are a helpful assistant answering questions about GitHub issues in a repository.
Use ONLY the issues provided below. Cite each issue you reference as [#N].
If the answer is not found in the provided issues, reply exactly:
"I couldn't find this in the indexed issues."\
"""


def answer(
    repo_slug: str,
    question: str,
    k: int = config.TOP_K_FINAL,
) -> dict:
    """
    Hybrid-retrieve relevant issues, build a cited prompt, and generate an answer.
    Returns {"answer": str, "sources": list[dict]}.
    """
    hits = retriever.hybrid_search(repo_slug, question, k_final=k)

    if not hits:
        return {"answer": "I couldn't find this in the indexed issues.", "sources": []}

    context_lines = []
    for h in hits:
        context_lines.append(
            f"[#{h['number']}] {h['title']} ({h['state']})\n{h['body_excerpt']}"
        )
    context = "\n\n".join(context_lines)

    prompt = f"{_SYSTEM}\n\nIssues:\n{context}\n\nQuestion: {question}\nAnswer:"
    raw = llm.generate(prompt)

    return {
        "answer": raw,
        "sources": [
            {"number": h["number"], "title": h["title"], "url": h["url"]}
            for h in hits
        ],
    }