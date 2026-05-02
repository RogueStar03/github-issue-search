import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

from src import indexer, retriever, rag, vector_store, state
from src.github_loader import slug


@click.group()
def cli():
    """github-issue-search — semantic search over GitHub issues. Fully local."""


@cli.command()
@click.option("--repo", required=True, help="GitHub repo in owner/name format.")
@click.option("--limit", default=None, type=int, help="Cap issue count (useful for rate-limited testing).")
def index(repo, limit):
    """Fetch and index issues from a GitHub repo."""
    click.echo(f"Indexing {repo}...")
    result = indexer.index_repo(repo, limit=limit)
    if result["new"] == 0:
        click.echo(f"No new issues. Total indexed: {result['total']}")
    else:
        click.echo(f"Indexed {result['new']} issue(s). Total: {result['total']}")


@cli.command()
@click.option("--repo", required=True, help="GitHub repo in owner/name format.")
@click.argument("query")
@click.option("--k", default=10, show_default=True, help="Number of results to return.")
def search(repo, query, k):
    """Hybrid search (vector + BM25) over indexed issues."""
    repo_slug = slug(repo)
    hits = retriever.hybrid_search(repo_slug, query, k_final=k)
    if not hits:
        click.echo("No results found. Make sure the repo is indexed.")
        return
    click.echo(f"\nTop {len(hits)} results for: \"{query}\"\n")
    for h in hits:
        v = h["vector_rank"]
        b = h["bm25_rank"]
        click.echo(
            f"  #{h['number']:5d}  [{h['state']:6s}]  score={h['rrf_score']:.4f}  "
            f"vec={v if v >= 0 else '-':>3}  bm25={b if b >= 0 else '-':>3}  "
            f"{h['title']}"
        )
        if h["url"]:
            click.echo(f"           {h['url']}")


@cli.command()
@click.option("--repo", required=True, help="GitHub repo in owner/name format.")
@click.argument("question")
@click.option("--k", default=10, show_default=True, help="Number of issues to retrieve for context.")
def ask(repo, question, k):
    """Ask a natural-language question over indexed issues (RAG)."""
    repo_slug = slug(repo)
    click.echo(f"\nSearching issues in {repo}...\n")
    result = rag.answer(repo_slug, question, k=k)
    click.echo(result["answer"])
    if result["sources"]:
        click.echo("\nSources:")
        for s in result["sources"]:
            click.echo(f"  [#{s['number']}] {s['title']}")
            if s["url"]:
                click.echo(f"         {s['url']}")


@cli.command("list")
def list_repos():
    """List all indexed repos and their issue counts."""
    repos = vector_store.list_repos()
    if not repos:
        click.echo("No repos indexed yet.")
        return
    click.echo("Indexed repos:")
    for repo_slug, count in repos:
        display = repo_slug.replace("__", "/")
        click.echo(f"  {display} — {count} issues")


@cli.command()
@click.option("--repo", default=None, help="Repo to clear (owner/name). Omit to clear all.")
@click.confirmation_option(prompt="This will delete indexed data. Are you sure?")
def clear(repo):
    """Remove indexed data from the vector store."""
    import shutil
    from src import config

    repo_slug = slug(repo) if repo else None
    vector_store.clear(repo_slug)

    # Also remove BM25 and state files
    if repo_slug:
        bm25_dir = config.BM25_PATH / repo_slug
        state_file = config.STATE_PATH / f"{repo_slug}.json"
        if bm25_dir.exists():
            shutil.rmtree(bm25_dir)
        if state_file.exists():
            state_file.unlink()
        click.echo(f"Cleared: {repo}")
    else:
        if config.BM25_PATH.exists():
            shutil.rmtree(config.BM25_PATH)
        if config.STATE_PATH.exists():
            shutil.rmtree(config.STATE_PATH)
        click.echo("All indexed data cleared.")


if __name__ == "__main__":
    cli()