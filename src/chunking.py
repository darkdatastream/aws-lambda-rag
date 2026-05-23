"""Document chunking with structural awareness.

Splits markdown/rst documents into chunks that respect document structure
(headings, paragraphs) rather than blindly cutting every N characters.

Strategy:
1. Split by markdown headings (H2 and below) when possible.
2. If a section is too large, fall back to splitting by paragraphs.
3. If a paragraph is still too large, fall back to fixed-size word splitting.
4. Apply overlap between adjacent chunks to preserve context across boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """A piece of a document with metadata for retrieval."""

    text: str
    source: str  # relative file path
    section: str  # nearest heading
    chunk_index: int  # position within source file

    def to_metadata(self) -> dict[str, str | int]:
        """Return metadata dict suitable for ChromaDB."""
        return {
            "source": self.source,
            "section": self.section,
            "chunk_index": self.chunk_index,
        }


# Match markdown headings: # Title, ## Subtitle, etc.
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Approximate tokens per word for English technical text
TOKENS_PER_WORD = 1.3


def _approx_token_count(text: str) -> int:
    """Rough token count estimate without loading a tokenizer."""
    return int(len(text.split()) * TOKENS_PER_WORD)


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, content) pairs.

    The first section may have an empty heading if the doc doesn't start with one.
    """
    if not text.strip():
        return []

    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []

    # Content before first heading
    if matches[0].start() > 0:
        intro = text[: matches[0].start()].strip()
        if intro:
            sections.append(("(intro)", intro))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((heading, content))

    return sections


def _split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Word-based splitter with overlap for sections that exceed max_tokens."""
    words = text.split()
    if not words:
        return []

    chunk_word_count = int(max_tokens / TOKENS_PER_WORD)
    overlap_word_count = int(overlap_tokens / TOKENS_PER_WORD)
    stride = max(chunk_word_count - overlap_word_count, 1)

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_word_count, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride

    return chunks


def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a document into chunks.

    Args:
        text: Raw document content (markdown or rst).
        source: Relative file path, used for citation.
        chunk_size: Target maximum tokens per chunk.
        chunk_overlap: Overlap tokens between adjacent chunks within a section.

    Returns:
        List of Chunk objects with text and metadata.
    """
    sections = _split_by_headings(text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for heading, content in sections:
        section_label = heading or "(no heading)"
        token_count = _approx_token_count(content)

        if token_count <= chunk_size:
            # Section fits in one chunk
            chunks.append(
                Chunk(
                    text=content,
                    source=source,
                    section=section_label,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        else:
            # Section too long, split with overlap
            for piece in _split_long_text(content, chunk_size, chunk_overlap):
                chunks.append(
                    Chunk(
                        text=piece,
                        source=source,
                        section=section_label,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    return chunks


def chunk_directory(
    directory: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    extensions: tuple[str, ...] = (".md", ".rst", ".txt"),
) -> list[Chunk]:
    """Walk a directory and chunk every document found.

    Args:
        directory: Root directory to scan recursively.
        chunk_size: Target tokens per chunk.
        chunk_overlap: Overlap tokens between adjacent chunks.
        extensions: File extensions to process.

    Returns:
        Flat list of all chunks from all documents.
    """
    all_chunks: list[Chunk] = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        if not text.strip():
            continue

        relative = path.relative_to(directory).as_posix()
        all_chunks.extend(
            chunk_document(text, source=relative, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )

    return all_chunks
