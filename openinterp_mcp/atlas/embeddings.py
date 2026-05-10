"""Text → embedding via OpenAI or Anthropic API.

Embed the (title + claim + model_id + tags) tuple per publication. We pick OpenAI's
text-embedding-3-small by default (1536d, $0.02 / 1M tokens). Anthropic doesn't yet ship
an embedding endpoint — when they do, swap providers here.
"""
from __future__ import annotations

import os
from typing import List


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def embed_text(text: str, model: str = DEFAULT_MODEL, provider: str = DEFAULT_PROVIDER) -> List[float]:
    if provider == "openai":
        return _openai_embed(text, model)
    raise ValueError(f"Unknown embedding provider: {provider}")


def embed_batch(texts: List[str], model: str = DEFAULT_MODEL, provider: str = DEFAULT_PROVIDER) -> List[List[float]]:
    if provider == "openai":
        return _openai_embed_batch(texts, model)
    raise ValueError(f"Unknown embedding provider: {provider}")


def _openai_embed(text: str, model: str) -> List[float]:
    return _openai_embed_batch([text], model)[0]


def _openai_embed_batch(texts: List[str], model: str) -> List[List[float]]:
    try:
        import openai
    except ImportError as e:
        raise RuntimeError("Embeddings need the openai SDK: `pip install openai`") from e
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Embeddings are opt-in.")
    client = openai.OpenAI()
    resp = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in resp.data]
