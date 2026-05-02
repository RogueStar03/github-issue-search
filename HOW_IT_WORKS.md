# How github-issue-search Works — A Plain-English Deep Dive

---

## The problem we are solving

You are about to file a bug on a GitHub repo. You type a title like "SSL certificate error on proxy."
But what if someone already reported this six months ago and it was closed with a fix?
You waste the maintainer's time and yours.

GitHub's built-in search is keyword-only. It finds "SSL" and "certificate" but not an issue
titled "HTTPS handshake fails behind corporate firewall" — which is the same problem, different words.

**We need meaning-aware search, not just word matching.**

But there is another edge case: you search for issue `#1234` or an exact error message like
`ConnectionError: ('Connection aborted.', RemoteDisconnected(...))`. A meaning-aware search is
terrible at these because error strings and numbers have no "meaning" — they are just symbols.
You need exact keyword matching for those.

**The real solution needs both.**

This project implements **hybrid search**: semantic (meaning) + BM25 (keyword), combined via a
ranking algorithm called RRF. That combination is what production search systems like Elasticsearch
and GitHub itself use under the hood.

---

## The two phases

```
PHASE 1 — INDEXING (run once per repo, then incrementally)
GitHub API → fetch issues → build text → embed to vectors → store in ChromaDB
                                      └─ also build BM25 keyword index

PHASE 2 — QUERYING (every time you search or ask)
Your query → semantic search (ChromaDB) ─┐
           → BM25 keyword search         ─┤─ RRF fusion → top-10 → (optionally) LLM answer
```

---

## Step 1 — Fetching issues from GitHub (`src/github_loader.py`)

**Goal:** Pull all issues from a public repo via the GitHub API.

**Library:** `PyGithub` — a Python wrapper around GitHub's REST API.

```python
from github import Github, Auth

gh = Github(auth=Auth.Token("your_token"))   # authenticated: 5000 req/hr
repo = gh.get_repo("psf/requests")
issues = repo.get_issues(state="all")        # all open + closed issues
```

**Why we need a token:**
Without authentication, GitHub allows only 60 API requests per hour per IP.
A repo like `psf/requests` has thousands of issues — you'd hit the limit in seconds.
With a Personal Access Token (PAT), the limit jumps to 5000/hr.
Even with no scopes selected, a PAT bumps the limit. We only need read access to public repos.

**Filtering out Pull Requests:**
GitHub's API has a quirk: every Pull Request is also an Issue internally.
When you call `get_issues()`, it returns both. We filter PRs with:

```python
if issue.pull_request is not None:
    continue   # skip — this is a PR, not a real issue
```

**Incremental sync — the `since` parameter:**
We do not re-download everything every time. After the first index run, we store
the timestamp of the last issue we saw. On the next run, we only fetch issues
updated after that timestamp:

```python
repo.get_issues(state="all", since=last_seen_datetime, sort="updated", direction="asc")
```

This means re-indexing a repo costs almost nothing if very few issues changed.
The timestamp is stored in `state/owner__name.json`.

**What goes in vs what comes out:**

```
INPUT:  "psf/requests"

OUTPUT:
[
  {
    "number": 6521,
    "title": "SSL certificate verification fails on corporate proxy",
    "body": "Getting SSLError when connecting through our corporate proxy...",
    "state": "open",
    "labels": "Bug,needs-triage",
    "created_at": "2024-11-03T14:22:00+00:00",
    "updated_at": "2024-11-05T09:10:00+00:00",
    "url": "https://github.com/psf/requests/issues/6521"
  },
  ...
]
```

---

## Step 2 — Building the text to embed (`src/indexer.py`)

**Goal:** Turn each issue dict into a single string we can embed.

We concatenate title and body:

```python
text = f"{issue['title']}\n\n{issue['body']}"
text = text[:1500]   # hard truncate
```

**Why not include comments?**
Each issue comment is a separate API call. A 100-issue repo with 10 comments each
= 1000 extra API calls. With 60/hr anonymous rate limit, that is 16 hours of waiting.
For v1, we skip comments and only embed the title + original body.

**Why truncate at 1500 characters?**
The `nomic-embed-text` model has a context window limit (roughly 2048 tokens).
1500 characters ≈ 375 tokens — well within the safe zone. Long issue bodies (with
code blocks, stack traces) are truncated at this point. The first 1500 characters
usually contain all the meaningful problem description anyway.

---

## Step 3 — Converting text to vectors ("embeddings") (`src/embeddings.py`)

*This is the same concept as in doc-assistant. A brief recap:*

**Goal:** Convert each issue's text into a list of 768 numbers that represents its *meaning*.

```python
import ollama

result = ollama.embed(model="nomic-embed-text", input=["SSL error on proxy", "login fails"])
vectors = result["embeddings"]
# → [[0.12, -0.45, ...], [0.08, -0.41, ...]]   ← two vectors, one per text
```

**Key insight:** Text with similar meaning produces similar numbers.

```
embed("SSL certificate error")          → [0.12, -0.45, 0.88, ...]
embed("HTTPS handshake fails")          → [0.11, -0.43, 0.85, ...]   ← very close!
embed("pagination not working on page 2") → [-0.78, 0.22, -0.34, ...]  ← far away
```

**Batching (an improvement over doc-assistant):**
The old doc-assistant calls Ollama once per text. We use the newer `ollama.embed(input=[...])`
which accepts a list — Ollama processes the whole batch in one call. 32 issues per batch
instead of 32 separate round trips to Ollama. Significantly faster for large repos.

---

## Step 4 — Storing in ChromaDB (`src/vector_store.py`)

**Goal:** Store all issue vectors in a database so we can query "find me the 50 issues
most similar to this question."

**One collection per repo:**
Unlike doc-assistant (one collection for all PDFs), here each repo gets its own collection.
The collection name is the repo slug: `psf/requests` → `psf__requests`.

```python
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("psf__requests")

collection.upsert(
    ids=["6521", "6522"],           # issue numbers as string IDs
    documents=["issue text...", "another issue..."],
    embeddings=[[0.12, ...], [0.08, ...]],
    metadatas=[
        {"title": "SSL error...", "state": "open", "url": "...", "labels": "Bug"},
        {"title": "Pagination...", "state": "closed", "url": "...", "labels": ""},
    ]
)
```

**`upsert` = insert or update:**
If issue #6521 already exists (from a previous index run), it gets overwritten.
If it is new, it gets inserted. This is how incremental re-indexing works without duplicates.

**Dedup by issue number:**
We use `str(issue.number)` as the ID. Run `index` twice on the same repo — second run
overwrites the same IDs. No duplicates, no separate hash tracking needed (simpler than
doc-assistant's SHA-256 approach).

---

## Step 5 — Building the BM25 keyword index (`src/bm25_index.py`)

**This is the new part compared to doc-assistant. Read carefully.**

### What is BM25?

BM25 is a keyword-based ranking algorithm. It has been the backbone of search engines
(including Google's early days) for 30+ years. It does NOT understand meaning.
It simply counts words.

Think of it like this: if you search for "proxy", BM25 counts how many times "proxy"
appears in each issue. But it is smarter than a raw word count — it uses two adjustments:

**Adjustment 1 — Term Frequency (TF):** An issue that mentions "proxy" 10 times is
probably more about proxies than one that mentions it once. But the relevance does not
scale linearly — going from 1 mention to 2 is a big jump; going from 10 to 11 barely matters.
BM25 applies a diminishing curve to this.

**Adjustment 2 — Inverse Document Frequency (IDF):** If the word "proxy" appears in
only 3 out of 100 issues, it is a rare, informative word — matching it means a lot.
If the word "error" appears in 90 out of 100 issues, matching it means almost nothing.
BM25 weights rare words higher and common words lower.

**BM25 final score for a document:**

```
For each word in the query:
  score += (how many times the word appears in this doc, with diminishing returns)
         × (how rare this word is across all docs)

Total score = sum of all word scores
```

In code (using the `bm25s` library):

```python
import bm25s

# Indexing — done once after fetching issues
corpus = ["SSL error on proxy", "login fails", "pagination broken", ...]
tokens = bm25s.tokenize(corpus, stopwords="en")   # split into words, remove "the", "a", "on"
retriever = bm25s.BM25()
retriever.index(tokens)
retriever.save("bm25_index/psf__requests/")        # save to disk

# Querying
query_tokens = bm25s.tokenize(["proxy SSL"])
results, scores = retriever.retrieve(query_tokens, k=50)
# results[0] → indices of top-50 matching documents (integers)
# scores[0]  → their BM25 scores
```

**Why do we save the index to disk?**
Unlike ChromaDB which auto-persists, `bm25s` is in-memory. If we did not save it,
we would have to rebuild it from scratch every time the app starts. Rebuilding is fast
for small repos but wasteful. After each `index` run we call `retriever.save(dir)`;
on the first query we call `BM25.load(dir)` to restore it.

**Why do we ALWAYS rebuild fully, never update incrementally?**
`bm25s` does not support "add these 10 new documents to the existing index." You have
to rebuild from all documents every time. This sounds wasteful but it is actually fast —
rebuilding a 1000-issue index takes under a second. The ChromaDB part handles the actual
storage; the BM25 index is just a search accelerator rebuilt from ChromaDB data:

```python
# In indexer.py, after upserting to ChromaDB:
all_data = vector_store.dump_all(slug)          # get ALL docs from Chroma
bm25_index.build(slug, all_data["ids"], all_data["documents"])  # rebuild BM25 from scratch
```

### When is BM25 better than semantic search?

| Query | Semantic | BM25 | Winner |
|---|---|---|---|
| "SSL error on proxy" | ✅ finds "HTTPS handshake fails" too | ✅ finds exact word matches | Tie |
| "login keeps failing" | ✅ finds "authentication broken" | ❌ misses if "login" not in issue | Semantic |
| `#1234` (issue number) | ❌ numbers have no meaning | ✅ finds exact string | BM25 |
| `ConnectionError: ('Connection aborted.')` | ❌ error string is not meaningful | ✅ exact match | BM25 |
| "how to set a timeout" | ✅ finds "configure max wait time" | ❌ different words | Semantic |
| "timeout" (single word) | ✅ finds "request hangs forever" | ✅ finds exact word | Tie |

**Neither is always better. That is why we use both.**

---

## Step 6 — Combining the two rankings with RRF (`src/retriever.py`)

**Goal:** We have two ranked lists of issues — one from semantic search, one from BM25.
We need to combine them into one final ranking.

### The problem with combining scores directly

Semantic search returns cosine distance scores (e.g. 0.05, 0.12, 0.18).
BM25 returns BM25 scores (e.g. 4.21, 3.18, 1.05).

These are completely different scales. You cannot just add them:
- A semantic score of 0.05 (very similar) looks small
- A BM25 score of 4.21 (good match) looks big
- Adding them: 0.05 + 4.21 = 4.26 — the semantic result barely contributes

You would need to normalize both to the same scale — but how? The distributions are different
for every query. Any normalization would involve guessing.

### RRF — Reciprocal Rank Fusion

RRF ignores the raw scores entirely. It only uses the **rank position** of each document.

The formula for a document's final score:

```
score = Σ  1 / (k + rank)
```

Where `k = 60` (a constant) and `rank` is the document's position in each list (0-indexed).
We sum this across both lists.

**Example with 3 documents:**

```
Semantic results:          BM25 results:
  Rank 0: issue #521         Rank 0: issue #100
  Rank 1: issue #100         Rank 1: issue #521
  Rank 2: issue #203         Rank 2: issue #999

RRF scores (k=60):
  issue #521: 1/(60+0) + 1/(60+1) = 0.01667 + 0.01639 = 0.03306  ← wins
  issue #100: 1/(60+1) + 1/(60+0) = 0.01639 + 0.01667 = 0.03306  ← tie
  issue #203: 1/(60+2) + 0        = 0.01613               ← only in semantic
  issue #999: 0        + 1/(60+2) = 0.01613               ← only in BM25
```

**Why k=60?**
The constant prevents the top rank from dominating too strongly. With k=0, rank 1 would
score 1.0 and rank 2 would score 0.5 — a huge gap. With k=60, rank 1 scores 1/61=0.0164
and rank 2 scores 1/62=0.0161 — nearly the same. This makes the fusion smoother and
rewards documents that appear in both lists more than documents that top only one.

**What the code does:**

```python
def _rrf(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:                       # two lists: [vec_ids], [bm25_ids]
        for rank, doc_id in enumerate(ranked):        # rank = position (0, 1, 2, ...)
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])  # sort by score, highest first
```

After RRF, we take the top 10 results and look up their full metadata from ChromaDB.

**The debug output you see in `cli search`:**
```
#6521  [open]  score=0.0325  vec=0  bm25=1  SSL certificate error on proxy
#6510  [closed] score=0.0315  vec=2  bm25=0  HTTPS handshake failure on corporate networks
```

`vec=0` means it was rank 0 (the top result) in semantic search.
`bm25=1` means it was rank 1 in BM25.
`score=0.0325` is the final RRF score (higher = better).
An issue with `bm25=-` only appeared in semantic results. One with `vec=-` only in BM25.

---

## Step 7 — Generating an answer with the LLM (`src/llm.py` + `src/rag.py`)

This is the "ask" part — you ask a question and get a written answer, not just a list.

**`llm.py` — talking to qwen2.5:7b**

```python
import ollama

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": prompt}],
    options={"num_ctx": 8192, "temperature": 0.2},
)
answer = response["message"]["content"]
```

`num_ctx: 8192` — qwen2.5:7b defaults to 2048 token context window in Ollama,
which is too small to fit 10 issues. We bump it to 8192 to fit all retrieved context.

`temperature: 0.2` — temperature controls how "creative" the model is.
At 0 it always picks the most likely next token (deterministic but boring).
At 1 it is creative but can go off-track. For RAG over factual issue data, 0.2
keeps it grounded while still producing natural sentences.

**`rag.py` — the prompt**

```
System: Answer using only the GitHub issues provided. Cite issues by number like [#42].
        If the answer isn't in the issues, say so.

Issues:
[#6521] SSL certificate verification fails on corporate proxy (open)
Getting SSLError when connecting through our corporate proxy server...

[#6510] HTTPS handshake failure on corporate networks (closed)
Unable to connect through Zscaler proxy, getting certificate verify failed...

[#6489] requests ignores REQUESTS_CA_BUNDLE env variable (closed)
...

Question: what SSL issues have been reported?
Answer:
```

**Why we need `num_ctx: 8192`:**
The prompt above contains 10 retrieved issues. Each issue contributes ~200-400 tokens of
context. 10 issues × 300 tokens = 3000 tokens of context, plus the question and
instructions. That alone exceeds the 2048 default. 8192 tokens is plenty.

**The full `ask` flow:**

```
rag.answer("what SSL issues have been reported?")
  ├── retriever.hybrid_search(...)     → top 10 issues (by RRF)
  ├── build context block              → "[#6521] title\nbody\n\n[#6510] ..."
  ├── assemble prompt                  → system + context + question
  ├── llm.generate(prompt)             → "Several SSL/certificate issues have been reported..."
  └── return {
        "answer": "Several SSL/certificate issues...",
        "sources": [{"number": 6521, "title": "...", "url": "..."}, ...]
      }
```

---

## Step 8 — Incremental sync and state (`src/state.py`)

**Goal:** On re-index runs, only fetch issues that changed since last time.

```python
# state/psf__requests.json
{
  "last_updated": "2024-11-05T09:10:00+00:00",
  "issue_count": 247
}
```

On the next `index` run:
1. Load `last_updated` from the JSON file
2. Pass it to `repo.get_issues(since=last_updated)`
3. GitHub returns only issues updated after that timestamp (could be 0 — no work to do)
4. Upsert those issues into ChromaDB (overwrites if they changed, inserts if they are new)
5. Rebuild BM25 from the now-updated ChromaDB corpus
6. Update the JSON with the new `last_updated` timestamp

**Why `last_updated` from the issues, not `datetime.now()`?**
If we store `datetime.now()` and the sync takes 5 minutes, any issue updated *during*
that 5 minutes might be missed (their timestamp falls between "when we started" and "now").
Using the maximum `updated_at` from the fetched issues means we never skip anything.

---

## Step 9 — The CLI (`cli.py`)

```
python cli.py index --repo psf/requests --limit 100   ← fetch 100 issues and index them
python cli.py search --repo psf/requests "SSL proxy"  ← hybrid search, print ranked list
python cli.py ask    --repo psf/requests "what SSL issues have been reported?"  ← RAG answer
python cli.py list                                     ← show all indexed repos and counts
python cli.py clear  --repo psf/requests               ← delete this repo's index
```

`--limit` is useful during testing (rate limit budgeting) or for large repos where
you only want a representative sample.

---

## Step 10 — The Gradio UI (`app.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  Tabs: [ Index ] [ Search & Ask ]                               │
└─────────────────────────────────────────────────────────────────┘

INDEX TAB:
┌─────────────────────────────────────────────────────────────────┐
│  Repo: [psf/requests        ]  Limit: [100]  [Index]            │
│  Status: Indexed 100 issues from psf/requests. Total: 100       │
└─────────────────────────────────────────────────────────────────┘

SEARCH & ASK TAB:
┌──────────────────────────────────────────────────────────────────┐
│  Repo: [psf/requests ▼]   k: [──●──── 10]                      │
│                                                                   │
│  Search: [SSL proxy error       ] [Hybrid Search]                │
│                                                                   │
│  Results:                                                         │
│    #6521 [open] score=0.0325 vec=0 bm25=1 SSL certificate error  │
│    #6510 [closed] score=0.0315 vec=2 bm25=0 HTTPS handshake...  │
│                                                                   │
│  ──────────────────────────────────────────────────────────────  │
│                                                                   │
│  [ Chat history ]                                                 │
│    You: what SSL issues have been reported?                       │
│    AI:  Several SSL/certificate issues have been reported...      │
│                                                                   │
│  Question: [                              ] [Ask]                 │
└──────────────────────────────────────────────────────────────────┘
```

The **Hybrid Search** button shows the ranked list with debug info (vec/bm25 ranks).
The **Ask** button runs the full RAG pipeline and shows a written answer with citations.

---

## Complete data flow — one full example

```
You run: python cli.py index --repo psf/requests --limit 100
│
├── github_loader.fetch_issues("psf/requests", since=None, limit=100)
│     ├── Auth with PAT → 5000 req/hr allowed
│     ├── iterate repo.get_issues(state="all") → skip PRs
│     └── return 100 issue dicts
│
├── indexer: build_doc() → truncate to 1500 chars → 100 text strings
│
├── embeddings.embed(100 texts, batch_size=32)
│     ├── batch 1: ollama.embed(input=[32 texts]) → 32 vectors of 768 numbers
│     ├── batch 2: ollama.embed(input=[32 texts]) → 32 vectors
│     ├── batch 3: ollama.embed(input=[32 texts]) → 32 vectors
│     └── batch 4: ollama.embed(input=[4 texts])  → 4 vectors
│         total: 100 vectors × 768 numbers each
│
├── vector_store.upsert("psf__requests", ids, docs, vectors, metadatas)
│     └── saved to chroma_db/psf__requests/
│
├── vector_store.dump_all("psf__requests") → all 100 docs for BM25
│
├── bm25_index.build("psf__requests", ids, docs)
│     ├── bm25s.tokenize(100 docs) → word lists, stopwords removed
│     ├── BM25().index(tokens)     → score matrix computed
│     └── save to bm25_index/psf__requests/
│
└── state.save("psf__requests", last_updated="2024-11-05T...", count=100)
    → state/psf__requests.json updated

You run: python cli.py search --repo psf/requests "SSL certificate proxy"
│
├── embeddings.embed(["SSL certificate proxy"])
│     └── → [0.12, -0.45, 0.88, ...]   (1 vector)
│
├── vector_store.query("psf__requests", vector, k=50)
│     └── ChromaDB finds 50 issues with most similar vectors
│         → [#6521, #6510, #6489, ...] (ranked by cosine similarity)
│
├── bm25_index.query("psf__requests", "SSL certificate proxy", k=50)
│     ├── tokenize query → ["ssl", "certificate", "proxy"]   (stopwords like "on" removed)
│     ├── BM25 scores each doc: rare words weighted more
│     └── → [(#6521, 4.21), (#100, 3.18), (#6510, 2.94), ...]
│
├── retriever._rrf([vec_ids, bm25_ids], k=60)
│     └── score = 1/(60+rank) from each list, summed
│         → [#6521: 0.0325, #6510: 0.0315, #6489: 0.0305, ...]
│
└── print top 10 with titles, ranks, URLs

You run: python cli.py ask --repo psf/requests "what SSL issues have been reported?"
│
├── (same hybrid search → top 10 issues)
│
├── rag.py: build context block
│     → "[#6521] SSL certificate fails (open)\nGetting SSLError...\n\n[#6510]..."
│
├── assemble prompt: system + context + question
│
├── ollama.chat("qwen2.5:7b", prompt, num_ctx=8192, temperature=0.2)
│     → "Several SSL/TLS issues have been reported. Issue [#6521] describes
│        certificate verification failures on corporate proxies..."
│
└── print answer + sources list
```

---

## Glossary

| Term | Plain English |
|---|---|
| **RAG** | Retrieval-Augmented Generation — find relevant pieces first, then generate an answer from only those pieces |
| **Embedding / Vector** | A list of numbers representing the *meaning* of text. Similar meaning → similar numbers |
| **Semantic search** | Search by meaning — finds issues about the same concept even if they use different words |
| **BM25** | A keyword ranking algorithm. Counts word frequency (weighted by how rare each word is). Used before AI search existed |
| **TF-IDF** | The older, simpler algorithm BM25 is based on. TF = how often a word appears in this doc. IDF = how rare it is across all docs |
| **Hybrid search** | Combining semantic + BM25. Catches both meaning matches and exact keyword/symbol matches |
| **RRF** | Reciprocal Rank Fusion — combines two ranked lists by position (rank) rather than raw scores, avoiding the scale mismatch problem |
| **k (in RRF)** | A constant (60) that softens the advantage of top-ranked results. Makes the fusion smoother |
| **ChromaDB** | A vector database. Stores vectors and can find the N most similar ones to a query vector |
| **Ollama** | A program that runs AI models locally on your machine. No internet needed after download |
| **nomic-embed-text** | A small AI model that ONLY converts text → embedding vector. Has no language abilities |
| **qwen2.5:7b** | A general-purpose LLM (7 billion parameters) that reads context and answers questions |
| **num_ctx** | The context window size for qwen2.5:7b. We set 8192 because the default 2048 is too small for RAG with 10 issues |
| **temperature** | How "creative" the LLM is. 0 = deterministic. 0.2 = slightly varied but mostly factual. 1.0 = creative/unpredictable |
| **PersistentClient** | ChromaDB mode where data is saved to disk. Survives restarts |
| **upsert** | Insert if new, overwrite if exists. How we handle re-indexing without creating duplicates |
| **Incremental sync** | Only fetch what changed since the last run, using the `since` timestamp parameter |
| **PAT** | Personal Access Token — a GitHub credential that bumps API rate limit from 60/hr to 5000/hr |
| **PaginatedList** | PyGithub type for large result sets. Lazily fetches the next page as you iterate — do NOT `.list()` it upfront |
| **bm25s** | The Python library we use for BM25. 100-500x faster than the older `rank_bm25` library |
| **stopwords** | Common words like "the", "a", "on" that carry no search meaning. BM25 ignores them |
| **click** | Python library for building CLI tools without manually parsing sys.argv |
| **gradio** | Python library for wrapping Python functions in a browser UI in ~20 lines |
