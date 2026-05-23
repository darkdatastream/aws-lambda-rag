"""End-to-end RAG pipeline: question -> hybrid retrieval -> generation -> answer.

This is the main entry point used by the CLI, Streamlit app, and tests.
Keep it thin - heavy lifting belongs in the component modules.
"""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from config import settings
from src.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from src.llm_client import LLMClient, get_llm_client
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.vector_store import RetrievedChunk


@dataclass
class RAGResponse:
    """Final response from the pipeline."""
    answer: str
    sources: list[RetrievedChunk]
    question: str


class RAGPipeline:
    """Orchestrates hybrid retrieval and generation for a single question."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.retriever = retriever or get_hybrid_retriever()
        self.llm = llm_client or get_llm_client()

    def ask(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Run the full pipeline for one question.

        Args:
            question: User's natural-language question.
            top_k: How many chunks to retrieve. Defaults to settings.top_k_retrieval.

        Returns:
            RAGResponse with answer text and the source chunks used.
        """
        k = top_k or settings.top_k_retrieval

        logger.debug(f"Question: {question}")

        # 1. Hybrid retrieval (vector + BM25 fused with RRF)
        chunks = self.retriever.retrieve(question, top_k=k)
        top_score = chunks[0].score if chunks else 0.0
        logger.info(f"Retrieved {len(chunks)} chunks (top RRF score: {top_score:.4f})")

        # 2. Build prompt and generate
        user_prompt = build_user_prompt(question, chunks)
        answer = self.llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)

        return RAGResponse(answer=answer, sources=chunks, question=question)
