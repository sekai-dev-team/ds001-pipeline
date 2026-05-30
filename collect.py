#!/usr/bin/env python3
"""DS-001 Pipeline — Passive information collection and ingestion.

Usage:
    python collect.py --mode daily
    python collect.py --mode hn-arxiv

Modes:
    daily       All 12 sources (RSS, X/Twitter, HN, arXiv)
    hn-arxiv    Hacker News + arXiv only (for the second daily run)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from pipeline.article import Article
from pipeline.sources import fetch_all, all_sources
from pipeline.llm_filter import filter_articles
from pipeline.knowledge_mcp import write_notes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("collect")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-001 Passive Information Pipeline"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["daily", "hn-arxiv"],
        help="Pipeline mode: daily (all sources) or hn-arxiv (HN + arXiv only)",
    )
    args = parser.parse_args()

    mode = args.mode
    pipeline_name = f"ds001-{mode}"
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("Starting pipeline: %s at %s", pipeline_name, timestamp)

    # ------------------------------------------------------------------
    # Step 1: Fetch all sources
    # ------------------------------------------------------------------
    source_results = fetch_all(mode)
    sources_total = len(source_results)
    sources_success = 0
    all_articles: list[Article] = []
    failures: list[str] = []

    for source_name, (articles, error) in source_results.items():
        if error:
            failures.append(f"{source_name}: {error}")
            logger.error("Source '%s' failed: %s", source_name, error)
        else:
            sources_success += 1
            all_articles.extend(articles)

    articles_fetched = len(all_articles)
    logger.info(
        "Fetched %d articles from %d/%d sources",
        articles_fetched,
        sources_success,
        sources_total,
    )

    # ------------------------------------------------------------------
    # Step 2: Deduplicate by URL
    # ------------------------------------------------------------------
    seen_urls: set[str] = set()
    unique_articles: list[Article] = []
    for article in all_articles:
        if article.url not in seen_urls:
            seen_urls.add(article.url)
            unique_articles.append(article)

    articles_deduped = len(unique_articles)
    logger.info(
        "Deduplicated: %d unique (removed %d duplicates)",
        articles_deduped,
        articles_fetched - articles_deduped,
    )

    # ------------------------------------------------------------------
    # Step 3: LLM filter
    # ------------------------------------------------------------------
    relevant_articles = filter_articles(unique_articles)
    articles_relevant = len(relevant_articles)
    logger.info("LLM filter: %d relevant articles", articles_relevant)

    # ------------------------------------------------------------------
    # Step 4: Write to knowledge-mcp
    # ------------------------------------------------------------------
    ingested, attempted = 0, 0
    if relevant_articles:
        ingested, attempted = write_notes(relevant_articles)
        logger.info("Ingested %d/%d notes into knowledge-mcp", ingested, attempted)
    else:
        logger.info("No relevant articles to ingest")

    # ------------------------------------------------------------------
    # Step 5: Print output statistics
    # ------------------------------------------------------------------
    stats = {
        "pipeline": pipeline_name,
        "timestamp": timestamp,
        "sources_total": sources_total,
        "sources_success": sources_success,
        "articles_fetched": articles_fetched,
        "articles_deduped": articles_deduped,
        "articles_relevant": articles_relevant,
        "articles_ingested": ingested,
        "failures": failures,
    }

    print()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    # Step 6: Exit code
    # ------------------------------------------------------------------
    # If ALL sources failed, exit non-zero so Hermes cron detects failure
    if sources_success == 0:
        logger.error("All sources failed — exiting with code 1")
        sys.exit(1)

    # If we have articles but none could be written, that's still non-fatal
    # (the LLM filter may have found nothing relevant)
    sys.exit(0)


if __name__ == "__main__":
    main()
