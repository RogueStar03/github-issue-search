import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from src import indexer, retriever, rag, vector_store
from src.github_loader import slug


def get_repo_choices() -> list[str]:
    return [r.replace("__", "/") for r, _ in vector_store.list_repos()]


def handle_index(repo: str, limit_str: str) -> tuple[str, gr.Dropdown]:
    repo = repo.strip()
    if not repo or "/" not in repo:
        return "Enter a repo in owner/name format.", gr.Dropdown()
    try:
        limit = int(limit_str) if limit_str.strip() else None
    except ValueError:
        return "Limit must be a number.", gr.Dropdown()

    result = indexer.index_repo(repo, limit=limit)
    if result["new"] == 0:
        msg = f"No new issues. Total indexed: {result['total']}"
    else:
        msg = f"Indexed {result['new']} issue(s) from **{repo}**. Total: {result['total']}"

    return msg, gr.Dropdown(choices=get_repo_choices(), value=repo)


def handle_search(repo: str, query: str, k: int) -> str:
    if not repo or not query.strip():
        return "Select a repo and enter a query."
    repo_slug = slug(repo)
    hits = retriever.hybrid_search(repo_slug, query, k_final=int(k))
    if not hits:
        return "No results found."
    lines = [f"**Top {len(hits)} results for:** \"{query}\"\n"]
    for h in hits:
        v = h["vector_rank"]
        b = h["bm25_rank"]
        v_str = str(v) if v >= 0 else "—"
        b_str = str(b) if b >= 0 else "—"
        url_part = f" · [{h['url']}]({h['url']})" if h["url"] else ""
        lines.append(
            f"**#{h['number']}** [{h['state']}] `score={h['rrf_score']:.4f}` "
            f"vec={v_str} bm25={b_str}{url_part}  \n"
            f"{h['title']}"
        )
    return "\n\n".join(lines)


def handle_ask(repo: str, question: str, history: list, k: int) -> tuple[list, str]:
    if not repo or not question.strip():
        return history, ""
    repo_slug = slug(repo)
    result = rag.answer(repo_slug, question, k=int(k))

    answer_text = result["answer"]
    if result["sources"]:
        citations = "  \n".join(
            f"[#{s['number']}] [{s['title']}]({s['url']})" for s in result["sources"]
        )
        answer_text += f"\n\n**Sources:**  \n{citations}"

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer_text})
    return history, ""


with gr.Blocks(title="GitHub Issue Search") as demo:
    gr.Markdown("# GitHub Issue Search\nSemantic + BM25 hybrid search over GitHub issues. Fully local.")

    with gr.Tabs():
        with gr.Tab("Index"):
            gr.Markdown("Fetch and index issues from any public GitHub repo.")
            with gr.Row():
                repo_input = gr.Textbox(label="Repo (owner/name)", placeholder="e.g. pallets/flask")
                limit_input = gr.Textbox(label="Issue limit (optional)", placeholder="e.g. 50")
            index_btn = gr.Button("Index", variant="primary")
            index_status = gr.Markdown("")

        with gr.Tab("Search & Ask"):
            repo_dropdown = gr.Dropdown(
                label="Select indexed repo",
                choices=get_repo_choices(),
                allow_custom_value=True,
            )
            k_slider = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Results (k)")

            with gr.Row():
                search_query = gr.Textbox(label="Search query", placeholder="login bug 401")
                search_btn = gr.Button("Hybrid Search", variant="secondary")
            search_results = gr.Markdown("")

            gr.Markdown("---")
            chatbot = gr.Chatbot(label="Ask a question", height=400)
            with gr.Row():
                question_box = gr.Textbox(label="Question", placeholder="What bugs have been reported about auth?")
                ask_btn = gr.Button("Ask", variant="primary")

    index_btn.click(
        fn=handle_index,
        inputs=[repo_input, limit_input],
        outputs=[index_status, repo_dropdown],
    )
    search_btn.click(
        fn=handle_search,
        inputs=[repo_dropdown, search_query, k_slider],
        outputs=search_results,
    )
    ask_btn.click(
        fn=handle_ask,
        inputs=[repo_dropdown, question_box, chatbot, k_slider],
        outputs=[chatbot, question_box],
    )
    question_box.submit(
        fn=handle_ask,
        inputs=[repo_dropdown, question_box, chatbot, k_slider],
        outputs=[chatbot, question_box],
    )

if __name__ == "__main__":
    demo.launch()