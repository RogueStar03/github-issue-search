# github-issue-search

Semantic + keyword hybrid search over GitHub issues — fully local, no paid API keys.

Ask plain-English questions across a repo's issue history, find duplicate bugs before filing, or search by exact error strings. Runs entirely on your machine using Ollama.

---

## What it does

- **Semantic search** — finds issues by meaning, even if they use different words ("login fails" matches "authentication broken")
- **BM25 keyword search** — finds exact matches for error strings, issue numbers, symbols
- **Hybrid (default)** — combines both via Reciprocal Rank Fusion for the best of both worlds
- **RAG Q&A** — ask a natural-language question, get a cited answer generated from retrieved issues
- **Incremental indexing** — re-run `index` anytime; only new/updated issues are fetched

---

## Stack

| Layer | Choice |
|---|---|
| Data source | GitHub REST API (PyGithub) |
| Embeddings | `nomic-embed-text` via Ollama (batched) |
| Vector DB | ChromaDB (local persistent) |
| Keyword search | `bm25s` — 100-500x faster than rank_bm25 |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| LLM | `qwen2.5:7b` via Ollama |
| CLI | click |
| UI | Gradio |

No cloud services. No API keys required for public repos (a GitHub PAT is recommended for rate limits).

---

## Prerequisites

**1. Ollama** — [download ollama.com](https://ollama.com), then pull the two models:

```powershell
ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull qwen2.5:7b         # ~4.7 GB — LLM for Q&A
```

**2. Python packages:**

```powershell
pip install chromadb ollama PyGithub bm25s click gradio python-dotenv
```

**3. (Recommended) GitHub Personal Access Token:**

Without a PAT, GitHub allows 60 API requests/hour — enough for a small test but not a real repo.
With a PAT (any scope, even none selected), the limit is 5000/hour.

Create one at [github.com/settings/tokens](https://github.com/settings/tokens) → Tokens (classic) → Generate new token.

Add it to a `.env` file in the project root:
```
GITHUB_TOKEN=ghp_your_token_here
```

`.env` is gitignored and Claude-ignored — it will not be committed or read by Claude.

---

## Usage

### Gradio UI (recommended)

```powershell
python app.py
```

Open [http://localhost:7860](http://localhost:7860).

- **Index tab** — enter a repo name (e.g. `psf/requests`), set a limit, click Index
- **Search & Ask tab** — select the repo, type a query, click Hybrid Search or Ask

### CLI

```powershell
# Index a repo (first run — fetches all issues up to --limit)
python cli.py index --repo psf/requests --limit 100

# Hybrid search — returns ranked list with semantic + BM25 ranks visible
python cli.py search --repo psf/requests "SSL certificate error"

# RAG Q&A — returns a written answer with issue citations
python cli.py ask --repo psf/requests "what SSL issues have been reported?"

# List all indexed repos and issue counts
python cli.py list

# Remove a repo's index
python cli.py clear --repo psf/requests

# Re-index (incremental — only fetches issues updated since last run)
python cli.py index --repo psf/requests
```

---

## How it works

See [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for a full plain-English walkthrough — covers embeddings, BM25, RRF fusion, incremental sync, and the complete data flow with examples.

Short version:

```
INDEX:
GitHub API → fetch issues → title + body text → embed (nomic-embed-text)
                                              → store in ChromaDB
                                              → build BM25 index (bm25s)

SEARCH:
query → semantic search (ChromaDB top-50)  ──┐
      → BM25 keyword search (bm25s top-50) ──┤── RRF fusion → top-10 results
                                              │
                              (optionally) → LLM answer (qwen2.5:7b)
```

---

## Project structure

```
github-issue-search/
├── src/
│   ├── config.py          # constants (models, paths, batch sizes, K values)
│   ├── github_loader.py   # PyGithub fetch + PR filter + since-checkpoint
│   ├── embeddings.py      # batched ollama.embed()
│   ├── vector_store.py    # ChromaDB client + upsert + query
│   ├── bm25_index.py      # bm25s build/save/load/query, lazy singleton
│   ├── retriever.py       # vector + BM25 → RRF fusion
│   ├── llm.py             # ollama.chat wrapper
│   ├── rag.py             # prompt + answer synthesis
│   ├── indexer.py         # orchestrates the full index pipeline
│   └── state.py           # last-updated checkpoint per repo (JSON)
├── cli.py                 # Click CLI
├── app.py                 # Gradio UI
├── HOW_IT_WORKS.md        # deep-dive explanation
├── chroma_db/             # vector store (gitignored)
├── bm25_index/            # BM25 index files (gitignored)
└── state/                 # sync checkpoints (gitignored)
```

---

## Notes

- **Single-repo per index** — each repo gets its own ChromaDB collection. Pass `--repo owner/name` to all commands.
- **Comments not included (v1)** — only title + body are indexed. Each comment would be a separate API call.
- **Body truncated at 1500 chars** — keeps within `nomic-embed-text`'s context window.
- **BM25 is rebuilt fully after every index run** — `bm25s` doesn't support incremental updates. Fast enough for repos up to ~10k issues.
