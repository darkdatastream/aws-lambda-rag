# AWS Lambda RAG Assistant

A small hybrid RAG demo for answering technical questions from selected AWS Lambda documentation pages.

The project combines semantic vector search, BM25 keyword retrieval, Reciprocal Rank Fusion, and an LLM answer layer with strict source grounding.

## What this project shows

This is a portfolio project showing a practical documentation assistant pipeline:

- fetch selected official AWS Lambda documentation pages as Markdown
- split documents into structured chunks
- embed chunks with a local sentence-transformers model
- store vectors in ChromaDB
- retrieve with hybrid search: vector search + BM25
- fuse results with Reciprocal Rank Fusion
- generate grounded answers with source citations
- expose the assistant through a simple Streamlit UI

## Dataset

The project currently indexes selected AWS Lambda Developer Guide Markdown pages:

- Lambda runtimes
- Lambda quotas
- function timeout
- function memory
- environment variables
- function configuration
- best practices
- concurrency
- scaling behavior

Downloaded documentation and the generated local vector database are excluded from git.

## Architecture

```text
AWS Lambda Markdown docs
        |
        v
fetch script
        |
        v
structured chunks
        |
        v
sentence-transformers embeddings
        |
        v
ChromaDB
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

## Stack

- Python
- ChromaDB
- sentence-transformers
- BAAI/bge-small-en-v1.5
- rank-bm25
- Reciprocal Rank Fusion
- DeepSeek API
- Streamlit

## Example questions

```text
What is the maximum Lambda function timeout?
What is the maximum memory for a Lambda function?
What Python runtimes does Lambda support?
How do I configure environment variables for a Lambda function?
What are the best practices for working with database connections in Lambda?
How does Lambda function scaling work?
```

## Example answer

Question:

```text
What is the maximum Lambda function timeout?
```

Answer:

```text
The maximum Lambda function timeout is 900 seconds (15 minutes)
[source: configuration-timeout.md]. This is also confirmed in the quotas table,
which lists "Function timeout" as "900 seconds (15 minutes)"
[source: gettingstarted-limits.md].
```

## Local setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e .
```

Create `.env`:

```bash
cp .env.example .env
```

Then add your DeepSeek API key:

```text
DEEPSEEK_API_KEY=your_key_here
```

## Build the local index

Fetch selected AWS Lambda Markdown documentation pages:

```bash
python scripts/01_fetch_docs.py --clean
```

Build the ChromaDB index:

```bash
python scripts/02_build_index.py --reset
```

Run a test question:

```bash
python scripts/03_test_retrieval.py --question "What is the maximum Lambda function timeout?"
```

Run the first five evaluation questions:

```bash
python scripts/03_test_retrieval.py --limit 5
```

## Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Repository hygiene

The following are excluded from git:

```text
.env
.venv/
data/
chroma_db/
*.log
__pycache__/
.pytest_cache/
```

## Known limitations

- The indexed documentation set is intentionally small.
- The assistant answers only from the downloaded Markdown pages.
- The local ChromaDB index must be rebuilt after changing the documentation set.
- API keys are required locally for the LLM generation step.

## Source attribution

This project indexes selected AWS Lambda documentation pages for a local RAG demo.

Downloaded documentation is stored locally under:

```text
data/docs/aws_lambda/
```

The fetch script also writes:

```text
data/docs/aws_lambda/SOURCE_ATTRIBUTION.md
```

The public repository excludes downloaded documentation by default. If documentation content is redistributed, the applicable AWS documentation terms must be respected.

## License

Project code: MIT.
