import ollama
from . import config


def embed(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per text. Uses batched ollama.embed() — faster than one-at-a-time."""
    out = []
    for i in range(0, len(texts), config.EMBED_BATCH_SIZE):
        batch = texts[i : i + config.EMBED_BATCH_SIZE]
        res = ollama.embed(model=config.EMBED_MODEL, input=batch)
        out.extend(res["embeddings"])
    return out