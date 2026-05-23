"""Step 2 of the pipeline: chunk documents, embed, and persist to ChromaDB.

Reads AWS Lambda Markdown docs, chunks each file, encodes batches with
sentence-transformers, and upserts into ChromaDB.

Run:
    python scripts/02_build_index.py
    python scripts/02_build_index.py --reset
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
from loguru import logger
from tqdm import tqdm

from config import settings
from src.chunking import chunk_directory
from src.embeddings import encode_texts
from src.vector_store import VectorStore


BATCH_SIZE = 64


@click.command()
@click.option("--reset", is_flag=True, help="Drop the existing collection before indexing.")
def main(reset: bool) -> None:
    """Build or rebuild the vector index."""
    if not settings.docs_dir.exists():
        logger.error(f"Docs directory not found: {settings.docs_dir}")
        logger.error("Run scripts/01_fetch_docs.py first.")
        sys.exit(1)

    logger.info(f"Chunking documents in {settings.docs_dir}...")
    chunks = chunk_directory(
        settings.docs_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        extensions=(".md",),
    )

    chunks = [chunk for chunk in chunks if chunk.source != "SOURCE_ATTRIBUTION.md"]
    logger.info(f"Produced {len(chunks)} chunks after filtering attribution file")

    if not chunks:
        logger.error("No chunks produced. Did fetch_docs.py succeed?")
        sys.exit(1)

    store = VectorStore()
    if reset:
        store.reset()

    logger.info(f"Encoding and indexing in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Indexing"):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [chunk.text for chunk in batch]
        embeddings = encode_texts(texts, batch_size=BATCH_SIZE, show_progress=False)
        store.add_chunks(batch, embeddings)

    logger.info(f"Index built. Total chunks: {store.count()}")
    logger.info("Next step: python scripts/03_test_retrieval.py --limit 3")


if __name__ == "__main__":
    main()
