"""Step 1 of the pipeline: fetch AWS Lambda documentation as Markdown.

The AWS documentation site exposes Markdown versions of selected pages.
This script downloads a focused set of official AWS Lambda Developer Guide pages.

Output:
    ./data/docs/aws_lambda/*.md

Run:
    python scripts/01_fetch_docs.py
    python scripts/01_fetch_docs.py --clean
"""

from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click
from loguru import logger

from config import settings


AWS_LAMBDA_PAGES: list[tuple[str, str]] = [
    ("lambda-runtimes", "https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.md"),
    ("gettingstarted-limits", "https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.md"),
    ("configuration-timeout", "https://docs.aws.amazon.com/lambda/latest/dg/configuration-timeout.md"),
    ("configuration-memory", "https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.md"),
    ("configuration-envvars", "https://docs.aws.amazon.com/lambda/latest/dg/configuration-envvars.md"),
    ("lambda-functions", "https://docs.aws.amazon.com/lambda/latest/dg/lambda-functions.md"),
    ("best-practices", "https://docs.aws.amazon.com/lambda/latest/dg/best-practices.md"),
    ("lambda-concurrency", "https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.md"),
    ("scaling-behavior", "https://docs.aws.amazon.com/lambda/latest/dg/scaling-behavior.md"),
]


def fetch_markdown(url: str) -> str:
    """Fetch one Markdown page from docs.aws.amazon.com."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"Unexpected HTTP status {response.status} for {url}")
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def write_attribution(docs_dir: Path) -> None:
    """Write source attribution required for a public documentation-derived demo."""
    retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# Source attribution",
        "",
        "This project indexes selected pages from the AWS Lambda Developer Guide.",
        "",
        "- Source: AWS Lambda Developer Guide",
        "- Format used: Markdown pages published by AWS documentation",
        "- Retrieved at: " + retrieved_at,
        "- AWS documentation licensing must be respected when redistributing derived content.",
        "",
        "Indexed source pages:",
        "",
    ]
    for slug, url in AWS_LAMBDA_PAGES:
        lines.append(f"- `{slug}.md` — {url}")
    (docs_dir / "SOURCE_ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@click.command()
@click.option("--clean", is_flag=True, help="Remove existing AWS Lambda docs before fetching.")
def main(clean: bool) -> None:
    """Fetch selected AWS Lambda documentation pages."""
    docs_dir: Path = settings.docs_dir

    if clean and docs_dir.exists():
        logger.info(f"Removing existing docs: {docs_dir}")
        shutil.rmtree(docs_dir)

    docs_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for slug, url in AWS_LAMBDA_PAGES:
        logger.info(f"Fetching {url}")
        try:
            markdown = fetch_markdown(url)
        except RuntimeError as exc:
            logger.error(str(exc))
            sys.exit(1)

        if not markdown.lstrip().startswith("#"):
            logger.warning(f"Fetched content does not look like Markdown heading: {url}")

        output = docs_dir / f"{slug}.md"
        header = (
            f"<!-- Source: {url} -->\n"
            f"<!-- Retrieved: {datetime.now(UTC).isoformat(timespec='seconds')} -->\n\n"
        )
        output.write_text(header + markdown.strip() + "\n", encoding="utf-8")
        total += 1

    write_attribution(docs_dir)

    logger.info(f"Done. Downloaded {total} AWS Lambda Markdown pages into {docs_dir}")
    logger.info("Next step: python scripts/02_build_index.py --reset")


if __name__ == "__main__":
    main()
