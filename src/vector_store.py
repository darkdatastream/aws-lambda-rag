"""ChromaDB vector store wrapper.

Thin abstraction over ChromaDB so the rest of the codebase doesn't depend
on Chroma's API surface directly. If you swap to Qdrant or pgvector later,
this is the only file you need to rewrite.

Persistence: ChromaDB writes to a directory on disk. The directory is the
"database file" — back it up, copy it between machines, delete to reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config import settings
from src.chunking import Chunk


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search."""

    text: str
    source: str
    section: str
    score: float  # cosine similarity (higher = more similar)
    page: int | None = None


class VectorStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, persist_dir: Path | None = None, collection_name: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.collection_name

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Use cosine distance to match our normalized embeddings
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"VectorStore ready: collection='{self.collection_name}', "
            f"existing_count={self.collection.count()}"
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Index a batch of chunks with pre-computed embeddings.

        Args:
            chunks: List of Chunk objects (text + metadata).
            embeddings: Pre-computed embeddings, parallel to chunks.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")

        if not chunks:
            return

        # ChromaDB requires unique IDs. Use source+chunk_index for stability
        # (re-running indexing will overwrite, not duplicate).
        ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(chunks)} chunks. Total now: {self.collection.count()}")

    def query(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve top-k most similar chunks for a query embedding.

        Args:
            query_embedding: Embedding vector of the query (same dim as indexed chunks).
            top_k: How many results to return.

        Returns:
            List of RetrievedChunk, ordered by similarity descending.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # ChromaDB returns distance (lower = closer). With cosine space,
        # similarity = 1 - distance.
        chunks: list[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=False,
        ):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=str(meta.get("source", "unknown")),
                    section=str(meta.get("section", "")),
                    page=int(meta["page"]) if "page" in meta else None,
                    score=float(1.0 - dist),
                )
            )
        return chunks

    def count(self) -> int:
        """Total number of indexed chunks."""
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection. Use when you want a clean rebuild."""
        logger.warning(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
