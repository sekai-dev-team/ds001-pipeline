"""LLM-based article filtering and summarization via DeepSeek API.

Uses DeepSeek V4 (deepseek-chat) with temperature 0.3 to determine
relevance and generate Chinese summaries.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

from pipeline.article import Article

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_ARTICLES_PER_BATCH = 20

SYSTEM_PROMPT = (
    "You are an AI news filter. For each article below, determine if it "
    "is relevant to: AI architecture, large language models, AI agents, "
    "open-source AI, or AI engineering paradigms. If relevant, generate a "
    "2-3 sentence Chinese summary. "
    "Return a JSON array of {relevant: bool, ai_summary: string}."
    "Always return valid JSON, one object per input article in the same order."
)


def _call_deepseek(articles: list[Article]) -> list[dict[str, Any]] | None:
    """Send a batch of articles to DeepSeek for filtering.

    Returns a list of dicts with ``relevant`` and ``ai_summary`` keys,
    or *None* on failure.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        return None

    article_texts = []
    for i, a in enumerate(articles, 1):
        article_texts.append(
            f"[{i}] Title: {a.title}\n    Summary: {a.summary[:300]}\n    URL: {a.url}"
        )

    user_content = (
        "Please analyze these articles:\n\n" + "\n---\n".join(article_texts)
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        raw = data["choices"][0]["message"]["content"]

        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (possibly with language hint)
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        results: list[dict[str, Any]] = json.loads(cleaned)
        return results

    except requests.exceptions.Timeout:
        logger.error("DeepSeek API timeout after 120s")
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("DeepSeek API request failed: %s", exc)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Failed to parse DeepSeek response: %s", exc)
        return None


def filter_articles(articles: list[Article]) -> list[Article]:
    """Filter and summarize articles using DeepSeek API.

    Articles are processed in batches of *MAX_ARTICLES_PER_BATCH*.
    Each article is updated in-place with ``relevant`` and ``ai_summary``.
    Only articles deemed relevant are returned.
    """
    if not articles:
        return []

    relevant_articles: list[Article] = []
    batches = [
        articles[i : i + MAX_ARTICLES_PER_BATCH]
        for i in range(0, len(articles), MAX_ARTICLES_PER_BATCH)
    ]

    for batch in batches:
        results = _call_deepseek(batch)
        if results is None:
            logger.warning("LLM filter failed for batch of %d articles, marking all as not relevant", len(batch))
            continue

        for article, result in zip(batch, results):
            article.relevant = bool(result.get("relevant", False))
            article.ai_summary = result.get("ai_summary", "")
            if article.relevant:
                relevant_articles.append(article)

    logger.info(
        "LLM filter: %d relevant out of %d articles",
        len(relevant_articles),
        len(articles),
    )
    return relevant_articles
