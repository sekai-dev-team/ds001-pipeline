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
import re
import sys
from datetime import datetime, timezone

from pipeline.article import Article
from pipeline.sources import fetch_all, all_sources
from pipeline.llm_filter import filter_articles
from pipeline.fulltext import fetch_fulltext, url_pattern_hint
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
    # Step 3.5: Fulltext extraction
    # ------------------------------------------------------------------
    fulltext_success = 0
    for article in relevant_articles:
        result = fetch_fulltext(article.url)
        if result is not None:
            article.fulltext = result
            article.has_fulltext = True
            fulltext_success += 1
        else:
            article.has_fulltext = False
            logger.warning("Fulltext extraction failed for: %s", article.url)

    total_relevant = len(relevant_articles)
    logger.info(
        "Fulltext: %d/%d extracted successfully",
        fulltext_success,
        total_relevant,
    )

    # ------------------------------------------------------------------
    # Step 3.75: Quality filtering (post-extraction thresholds)
    # ------------------------------------------------------------------
    quality_articles: list[Article] = []
    quality_rejected = 0
    MIN_FULLTEXT_CHARS = 800
    MIN_DISCUSSION_FULLTEXT_CHARS = 400
    MIN_SUMMARY_CHARS = 200

    for article in relevant_articles:
        url_hint = url_pattern_hint(article.url)

        if article.has_fulltext:
            min_chars = (
                MIN_DISCUSSION_FULLTEXT_CHARS
                if article.content_type == "discussion"
                else MIN_FULLTEXT_CHARS
            )
            if article.fulltext and len(article.fulltext) >= min_chars:
                quality_articles.append(article)
            else:
                text_len = len(article.fulltext) if article.fulltext else 0
                logger.warning(
                    "Quality reject: %s (fulltext too short: %d chars, min %d; url_hint=%s, content_type=%s)",
                    article.title, text_len, min_chars, url_hint, article.content_type,
                )
                quality_rejected += 1
        else:
            # Tweets use RSS summary as the acceptable fallback content
            # (nitter pages are JS-rendered, so fulltext extraction is skipped)
            if url_hint == "tweet":
                quality_articles.append(article)
                logger.debug(
                    "Tweet fallback accepted: %s (summary=%d chars; url_hint=%s)",
                    article.title, len(article.summary) if article.summary else 0, url_hint,
                )
            elif article.summary and len(re.sub(r'<[^>]+>', '', article.summary).strip()) >= MIN_SUMMARY_CHARS:
                quality_articles.append(article)
            else:
                summary_len = len(article.summary) if article.summary else 0
                logger.warning(
                    "Quality reject: %s (no fulltext, summary too short: %d chars, min %d; url_hint=%s)",
                    article.title, summary_len, MIN_SUMMARY_CHARS, url_hint,
                )
                quality_rejected += 1

    logger.info(
        "Quality filter: %d passed, %d rejected (min %d chars fulltext / "
        "%d chars discussion / %d chars summary)",
        len(quality_articles), quality_rejected,
        MIN_FULLTEXT_CHARS, MIN_DISCUSSION_FULLTEXT_CHARS, MIN_SUMMARY_CHARS,
    )

    # ------------------------------------------------------------------
    # Step 4: Write to knowledge-mcp
    # ------------------------------------------------------------------
    ingested, attempted = 0, 0
    if quality_articles:
        ingested, attempted = write_notes(quality_articles)
        logger.info("Ingested %d/%d notes into knowledge-mcp", ingested, attempted)
    else:
        logger.info("No quality-passing articles to ingest")

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
