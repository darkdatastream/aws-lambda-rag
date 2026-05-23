"""Prompt templates for RAG generation.

The system prompt enforces three things:
1. Grounding: answer ONLY from provided context, never from training data.
2. Honesty: if context does not contain the answer, say so explicitly.
3. Citation: every factual claim must reference the retrieved source file.
"""

from __future__ import annotations

from src.vector_store import RetrievedChunk


SYSTEM_PROMPT = """You are a technical assistant for AWS Lambda documentation.

Strict rules:
1. Answer ONLY using the provided context below. Do not use prior knowledge.
2. If the context does not contain the answer, reply exactly: "I don't have enough information in the provided documentation to answer that."
3. Cite every factual claim using [source: filename.md], where filename.md comes from the chunk header.
4. Use concise technical prose. Show commands or code examples only when the context provides them.
5. Never invent limits, runtime names, configuration steps, quotas, API behavior, or best practices. If unsure, say so explicitly."""


def _format_chunk_header(chunk: RetrievedChunk, idx: int) -> str:
    """Build a human-readable header that the model can cite from."""
    parts = [f"Chunk {idx}"]
    if chunk.source and chunk.source != "unknown":
        parts.append(f"source: {chunk.source}")
    if chunk.section:
        parts.append(f"section: {chunk.section}")
    return f"[{' | '.join(parts)}]"


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context block for the user prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = _format_chunk_header(chunk, i)
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the final user message: context + question."""
    if not chunks:
        return (
            "Context: (no relevant documentation found)\n\n"
            f"Question: {question}"
        )

    context = format_context(chunks)
    return f"""Context retrieved from AWS Lambda documentation:

{context}

---

Question: {question}

Answer using only the context above. Cite sources inline as [source: filename.md]."""
