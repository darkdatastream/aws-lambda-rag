# Architecture

This project is a compact hybrid RAG pipeline for selected AWS Lambda documentation pages.

## Pipeline

```text
AWS Lambda Markdown docs
        |
        v
scripts/01_fetch_docs.py
        |
        v
data/docs/aws_lambda/*.md
        |
        v
src/chunking.py
        |
        v
sentence-transformers embeddings
        |
        v
ChromaDB persistent vector store
        |
        +----------------------+
        |                      |
        v                      v
vector search              BM25 search
        |                      |
        +----------+-----------+
                   v
        Reciprocal Rank Fusion
                   |
                   v
        grounded prompt context
                   |
                   v
              DeepSeek LLM
                   |
                   v
        cited technical answer
```

## Main components

- `scripts/01_fetch_docs.py` downloads selected AWS Lambda Markdown documentation pages.
- `scripts/02_build_index.py` chunks and embeds the downloaded documents.
- `src/vector_store.py` wraps ChromaDB persistence and similarity search.
- `src/hybrid_retriever.py` combines vector retrieval and BM25 using Reciprocal Rank Fusion.
- `src/prompts.py` builds grounded prompts with source citations.
- `src/rag_pipeline.py` connects retrieval and generation.
- `app/streamlit_app.py` provides the demo UI.
