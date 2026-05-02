import ollama
from . import config


def generate(prompt: str) -> str:
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": config.NUM_CTX, "temperature": 0.2},
    )
    return response["message"]["content"].strip()