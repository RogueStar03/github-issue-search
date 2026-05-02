# github-issue-search

Semantic + BM25 hybrid search over GitHub issues. Fully local — no API keys required for public repos.

## Vault context
For current project state, read:
`D:\Vaults\DevBrain\Projects\rag-demos\Current-State.md`

## Stack
| Layer | Choice |
|-------|--------|
| Data source | GitHub REST API via PyGithub |
| Embeddings | nomic-embed-text via Ollama (batched) |
| Vector DB | chromadb (local persistent) |
| Keyword search | bm25s |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| LLM | qwen2.5:7b via Ollama |
| CLI | click |
| UI | gradio |

## Hard rules
- Keep fully local — no paid API keys, no cloud services
- One ChromaDB collection per repo (slug: owner__name)
- BM25 index must be rebuilt fully after every index run (bm25s doesn't support incremental updates)
- Issue IDs in ChromaDB are str(issue.number) — upsert is the dedup mechanism
- PRs must be filtered out (issue.pull_request is not None → skip)

## Commands
```powershell
python cli.py index --repo owner/name [--limit N]
python cli.py search --repo owner/name "query"
python cli.py ask --repo owner/name "question"
python cli.py list
python cli.py clear [--repo owner/name]
python app.py   # Gradio UI
```

## Last Session

**2026-05-02 — Built github-issue-search from scratch: hybrid RAG CLI + Gradio UI**

### What changed
- Created full project: `src/` (config, state, github_loader, embeddings, vector_store, bm25_index, retriever, llm, rag, indexer)
- `cli.py` — Click CLI with index, search, ask, list, clear commands
- `app.py` — Gradio UI with Index tab + Search & Ask tab
- `HOW_IT_WORKS.md` — deep-dive doc explaining BM25, RRF, and the full pipeline
- `.env` + `.claudeignore` + `python-dotenv` wired for GitHub PAT
- Fixed BM25 query bug: `r.corpus` is None after mmap load — switched to index-based ID lookup
- Reduced `MAX_BODY_CHARS` from 6000 → 1500 to stay within nomic-embed-text context limit

### Key decisions
- Single-repo per Chroma collection (slug: owner__name); upsert by issue number for dedup
- BM25 always rebuilt fully from ChromaDB corpus after each index run (bm25s has no incremental update)
- Comments skipped in v1 — each is a separate API call, blows anonymous rate limit

### Next steps
- [ ] Test index on a real repo (e.g. `psf/requests --limit 100`) with PAT set
- [ ] Smoke-test Gradio UI end-to-end (index → search → ask in browser)
- [ ] Add comments fetching behind `--with-comments` flag for richer context