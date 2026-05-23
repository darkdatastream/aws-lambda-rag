"""Embedding model wrapper.

Loads the sentence-transformers model lazily (first use) and exposes a simple
encode() interface. The model is small (~80 MB) and runs on CPU comfortably
for batches up to a few thousand chunks.

Default model: BAAI/bge-small-en-v1.5
  - 384-dimensional embeddings
  - English-focused but handles code identifiers well
  - Strong performance on retrieval benchmarks (MTEB) for its size
"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger
from sentence_transformers import SentenceTransformer

from config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Return a cached SentenceTransformer instance.

    First call downloads the model (~80 MB) and may take 10-30s.
    Subsequent calls return the cached instance instantly.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    logger.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def encode_texts(texts: list[str], batch_size: int = 32, show_progress: bool = True) -> list[list[float]]:
    """Encode a list of texts into embedding vectors.

    Args:
        texts: List of strings to embed.
        batch_size: How many texts to embed at once. Lower if you run out of RAM.
        show_progress: Show a tqdm progress bar (useful for large batches).

    Returns:
        List of float vectors. Each inner list has `embedding_dim` elements.
    """
    if not texts:
        return []

    model = get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity becomes a dot product
    )
    return embeddings.tolist()


def encode_query(query: str) -> list[float]:
    """Encode a single query string. Convenience wrapper for retrieval."""
    return encode_texts([query], show_progress=False)[0]
