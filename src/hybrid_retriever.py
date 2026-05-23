"""Hybrid retrieval combining vector search (semantic) and BM25 (keyword).

Why hybrid? Vector embeddings are strong at semantic similarity but weak at
matching exact tokens (register names, numeric values, acronyms). BM25 is the
opposite. Fusing both retrievers with Reciprocal Rank Fusion (RRF, Cormack 2009)
consistently outperforms either alone on technical documentation.

The BM25 index is built in memory from all chunks in ChromaDB at first use.
For a 762-chunk corpus this takes <1s and uses ~1MB of RAM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from loguru import logger
from rank_bm25 import BM25Okapi

from config import settings
from src.embeddings import encode_query
from src.vector_store import RetrievedChunk, VectorStore


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Good enough for English technical text."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _ChunkCorpus:
    """All chunks loaded from ChromaDB, ready for BM25 indexing."""
    texts: list[str]
    metadatas: list[dict]
    ids: list[str]


class HybridRetriever:
    """Combines vector search and BM25 with Reciprocal Rank Fusion."""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()
        self._bm25: BM25Okapi | None = None
        self._corpus: _ChunkCorpus | None = None

    def _build_bm25(self) -> None:
        """Load all chunks from ChromaDB and build the BM25 index in memory."""
        logger.info("Building BM25 index from ChromaDB...")
        all_data = self.vector_store.collection.get(include=["documents", "metadatas"])
        texts = all_data["documents"]
        metadatas = all_data["metadatas"]
        ids = all_data["ids"]

        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._corpus = _ChunkCorpus(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"BM25 index ready: {len(texts)} documents indexed")

    def _ensure_bm25(self) -> None:
        if self._bm25 is None:
            self._build_bm25()

    def _vector_search(self, query: str, top_k: int) -> list[tuple[str, RetrievedChunk]]:
        """Return list of (chunk_id, RetrievedChunk) for top_k vector hits."""
        q_emb = encode_query(query)
        results = self.vector_store.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        out: list[tuple[str, RetrievedChunk]] = []
        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=False,
        ):
            chunk = RetrievedChunk(
                text=doc,
                source=str(meta.get("source", "unknown")),
                section=str(meta.get("section", "")),
                score=float(1.0 - dist),
                page=int(meta["page"]) if "page" in meta else None,
            )
            out.append((cid, chunk))
        return out

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, RetrievedChunk]]:
        """Return list of (chunk_id, RetrievedChunk) for top_k BM25 hits."""
        assert self._bm25 is not None and self._corpus is not None
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # Get top_k indices by score
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        out: list[tuple[str, RetrievedChunk]] = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            meta = self._corpus.metadatas[idx]
            chunk = RetrievedChunk(
                text=self._corpus.texts[idx],
                source=str(meta.get("source", "unknown")),
                section=str(meta.get("section", "")),
                score=float(scores[idx]),
                page=int(meta["page"]) if "page" in meta else None,
            )
            out.append((self._corpus.ids[idx], chunk))
        return out

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Run both retrievers, fuse with RRF, return top_k merged chunks.

        Args:
            query: Natural-language query.
            top_k: Final number of chunks to return (default: settings.top_k_retrieval).

        Returns:
            List of RetrievedChunk, ordered by RRF score (highest first).
            The .score field is replaced with the RRF score for transparency.
        """
        self._ensure_bm25()
        final_k = top_k or settings.top_k_retrieval
        rrf_k = settings.rrf_k

        vector_hits = self._vector_search(query, settings.vector_top_k)
        bm25_hits = self._bm25_search(query, settings.bm25_top_k)

        # RRF: score(d) = sum over retrievers of 1 / (rrf_k + rank_in_retriever)
        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, RetrievedChunk] = {}

        for rank, (cid, chunk) in enumerate(vector_hits):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            chunk_by_id[cid] = chunk

        for rank, (cid, chunk) in enumerate(bm25_hits):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            # Prefer vector chunk if already present (it has cosine score we may want later)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = chunk

        # Sort by RRF score and return top_k
        ranked_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:final_k]

        out: list[RetrievedChunk] = []
        for cid in ranked_ids:
            chunk = chunk_by_id[cid]
            # Replace score field with normalized RRF score for display
            chunk.score = rrf_scores[cid]
            out.append(chunk)

        logger.info(
            f"Hybrid retrieval: {len(vector_hits)} vector + {len(bm25_hits)} BM25 "
            f"-> {len(out)} fused (top RRF score: {out[0].score:.4f})"
            if out else "Hybrid retrieval: no results"
        )
        return out


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    """Cached singleton — BM25 index is built once per process."""
    return HybridRetriever()
