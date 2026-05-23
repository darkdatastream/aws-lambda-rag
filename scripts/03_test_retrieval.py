"""Step 3 of the pipeline: smoke test the full RAG pipeline.

Loads test questions from eval/test_questions.yaml, runs them through
the pipeline, and prints answers with source citations.

Run:
    python scripts/03_test_retrieval.py
    python scripts/03_test_retrieval.py --limit 3
    python scripts/03_test_retrieval.py --question "What is the maximum Lambda timeout?"
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click
import yaml
from loguru import logger

from src.rag_pipeline import RAGPipeline


def load_questions(path: Path) -> list[str]:
    """Load test questions from a YAML file."""
    if not path.exists():
        logger.error(f"Questions file not found: {path}")
        sys.exit(1)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [item["question"] for item in data.get("questions", [])]


def print_response(response) -> None:
    """Pretty-print a RAGResponse to the terminal."""
    print("\n" + "=" * 80)
    print(f"Q: {response.question}")
    print("=" * 80)
    print(f"\n{response.answer}\n")
    print("-" * 80)
    print("Sources used:")
    for i, src in enumerate(response.sources, start=1):
        print(f"  [{i}] {src.source} :: {src.section} (score={src.score:.4f})")
    print()


@click.command()
@click.option("--question", "-q", default=None, help="Ask a single ad-hoc question.")
@click.option("--limit", "-n", default=None, type=int, help="Run only the first N test questions.")
@click.option(
    "--questions-file",
    default="eval/test_questions.yaml",
    type=click.Path(path_type=Path),
    help="Path to test questions YAML.",
)
def main(question: str | None, limit: int | None, questions_file: Path) -> None:
    """Run the RAG pipeline against test questions or one ad-hoc query."""
    pipeline = RAGPipeline()

    if question:
        questions = [question]
    else:
        questions = load_questions(questions_file)
        if limit:
            questions = questions[:limit]

    logger.info(f"Running {len(questions)} question(s)")

    for q in questions:
        response = pipeline.ask(q)
        print_response(response)


if __name__ == "__main__":
    main()
