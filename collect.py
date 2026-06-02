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
import os
import re
import sys
from datetime import datetime, timezone

from pipeline.article import Article
from pipeline.sources import fetch_all, all_sources
from pipeline.llm_filter import filter_articles
from pipeline.fulltext import url_pattern_hint
from pipeline.knowledge_mcp import write_notes_fs, reindex_vault, write_digest, process_tags, _read_tag_library
from pipeline.digest import generate_digest

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
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    tag_library = _read_tag_library()
    relevant_articles = filter_articles(unique_articles, api_key, tag_library)
    articles_relevant = len(relevant_articles)
    logger.info("LLM filter: %d relevant articles", articles_relevant)

    # ------------------------------------------------------------------
    # Step 3.1: Tag processing (dedup + tag library management)
    # ------------------------------------------------------------------
    if relevant_articles:
        process_tags(relevant_articles)
        logger.info("Tag processing complete for %d articles", len(relevant_articles))

    # ------------------------------------------------------------------
    # Step 3.25: Section marker validation (v0.4.2)
    # ------------------------------------------------------------------
    # v0.5: match both standalone markers (方法:, 发现:) and combined (方法/发现:)
    SECTION_MARKER_RE = re.compile(r'\*\*(?:主题|方法(?:/发现)?|发现|意义):\*\*')
    validated_articles: list[Article] = []
    for article in relevant_articles:
        if SECTION_MARKER_RE.search(article.ai_summary):
            validated_articles.append(article)
        else:
            logger.warning(
                "Section-marker reject: %s (ai_summary lacks **主题:**, "
                "**方法:**, **发现:**, or **意义:** markers)",
                article.title,
            )
    rejected_markers = articles_relevant - len(validated_articles)
    if rejected_markers:
        logger.warning(
            "Section-marker filter: %d/%d articles rejected for missing "
            "required markers", rejected_markers, articles_relevant,
        )
    relevant_articles = validated_articles

    # ------------------------------------------------------------------
    # Step 3.75: Quality filtering (post-extraction thresholds)
    # ------------------------------------------------------------------
    # (Fulltext extraction now happens inside filter_articles() between
    # Pass 1 and Pass 2.  article.fulltext and article.has_fulltext are
    # already populated.)
    quality_articles: list[Article] = []
    quality_rejected = 0
    MIN_SUMMARY_CHARS = 150  # was 50; raised to 150 per v0.4.2 spec

    for article in relevant_articles:
        url_hint = url_pattern_hint(article.url)

        if article.has_fulltext:
            # Accept ALL extracted fulltext — don't reject based on length
            if article.fulltext and len(article.fulltext.strip()) > 0:
                quality_articles.append(article)
            else:
                logger.warning("Quality reject: %s (fulltext is empty)", article.title)
                quality_rejected += 1
        else:
            # Tweets: always accept (RSS summary IS the content)
            if url_hint == "tweet":
                quality_articles.append(article)
            # Other: accept if summary exists (lowered threshold to 50 chars)
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
        "Quality filter: %d passed, %d rejected (min %d chars summary for non-tweet, no-fulltext articles)",
        len(quality_articles), quality_rejected, MIN_SUMMARY_CHARS,
    )

    # ------------------------------------------------------------------
    # Step 4: Write articles to vault filesystem (filesystem-first)
    # ------------------------------------------------------------------
    ingested, attempted = 0, 0
    if quality_articles:
        ingested, attempted = write_notes_fs(quality_articles)
        logger.info("Wrote %d/%d notes to vault filesystem", ingested, attempted)

        # Bulk reindex once after all files are written
        if ingested > 0:
            logger.info("Triggering bulk vault reindex...")
            if reindex_vault():
                logger.info("Vault reindex complete")
            else:
                logger.warning("Vault reindex failed — notes are on disk but not searchable")
    else:
        logger.info("No quality-passing articles to ingest")

    # ------------------------------------------------------------------
    # Step 4.25: Generate daily digest
    # ------------------------------------------------------------------
    from collections import defaultdict

    per_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"fetched": 0, "relevant": 0, "fulltext": 0, "quality_passed": 0, "ingested": 0}
    )

    for article in all_articles:
        per_source[article.source_name]["fetched"] += 1

    for article in relevant_articles:
        per_source[article.source_name]["relevant"] += 1
        if article.has_fulltext:
            per_source[article.source_name]["fulltext"] += 1

    for article in quality_articles:
        per_source[article.source_name]["quality_passed"] += 1

    # For ingested — track by article in write_notes_fs
    # (write_notes_fs writes in order, so match by index)
    for article in quality_articles[:ingested]:
        per_source[article.source_name]["ingested"] += 1

    ingested_articles = quality_articles[:ingested] if quality_articles else []
    if ingested_articles:
        logger.info(
            "Generating daily digest for %d ingested articles...",
            len(ingested_articles),
        )
        digest_md = generate_digest(ingested_articles, mode, timestamp, per_source=per_source)
        if write_digest(digest_md, timestamp, mode=mode):
            logger.info("Digest saved to vault")
        else:
            logger.warning("Digest could not be saved to vault")
    else:
        logger.info("No ingested articles to digest")

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
        "per_source": {
            name: dict(counts)
            for name, counts in per_source.items()
        },
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
