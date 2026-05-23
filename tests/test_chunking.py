"""Unit tests for src/chunking.py."""

from __future__ import annotations

from src.chunking import chunk_document


def test_short_document_produces_one_chunk():
    text = "# Intro\n\nShort content."
    chunks = chunk_document(text, source="test.md")
    assert len(chunks) == 1
    assert chunks[0].source == "test.md"
    assert chunks[0].section == "Intro"


def test_multiple_headings_produce_multiple_chunks():
    text = """# Section A

Content A.

# Section B

Content B.

# Section C

Content C.
"""
    chunks = chunk_document(text, source="multi.md")
    assert len(chunks) == 3
    assert [c.section for c in chunks] == ["Section A", "Section B", "Section C"]


def test_long_section_gets_split():
    # 2000-word section should overflow chunk_size=200
    words = " ".join(f"word{i}" for i in range(2000))
    text = f"# Big section\n\n{words}"
    chunks = chunk_document(text, source="big.md", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.section == "Big section" for c in chunks)


def test_chunk_indices_are_sequential():
    text = "# A\n\nFirst.\n\n# B\n\nSecond.\n\n# C\n\nThird."
    chunks = chunk_document(text, source="seq.md")
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_empty_document_produces_no_chunks():
    chunks = chunk_document("", source="empty.md")
    assert chunks == []
