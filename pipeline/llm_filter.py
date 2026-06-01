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
MAX_ARTICLES_PER_BATCH = 50

SYSTEM_PROMPT = (
    "You are an AI news filter. For each article below:\n"
    "1. Classify its content type:\n"
    '   - "article": substantive blog, news, research write-up -- the target\n'
    '   - "discussion": HN thread, Reddit post, forum -- may still be relevant\n'
    "     if it contains substantive professional insight (not just jokes,\n"
    "     one-liners, or vote counts)\n"
    '   - "link_only": just points to external URL with no useful summary\n'
    '   - "low_quality": requires JS, paywall, just metadata or bibtex\n'
    "2. Determine if it is relevant to: AI architecture, large language models, "
    "AI agents, open-source AI, or AI engineering paradigms.\n"
    "   Both ``article`` and ``discussion`` content types can be relevant.\n"
    "   Only ``link_only`` and ``low_quality`` should be rejected outright.\n"
    "3. If relevant, generate a Chinese summary with these structured sections:\n"
    "   **主题:** [1 sentence — what the article is about]\n"
    "   **方法/发现:** [2-3 sentences — key method, result, or insight]\n"
    "   **意义:** [1 sentence — why it matters]\n"
    "   Total: at least 150 Chinese characters, at most 400.\n"
    'Return a JSON array of {{relevant: bool, ai_summary: string, content_type: string}}.\n'
    "Always return valid JSON, one object per input article in the same order."
)


def _call_deepseek(articles: list[Article]) -> list[dict[str, Any]] | None:
    """Send a batch of articles to DeepSeek for filtering.

    Returns a list of dicts with ``relevant``, ``ai_summary``, and
    ``content_type`` keys, or *None* on failure.
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
    Each article is updated in-place with ``relevant``, ``ai_summary``,
    and ``content_type``.
    Only articles deemed relevant AND classified as "article" or "discussion"
    content type are returned. ``link_only`` and ``low_quality`` are rejected.
    """
    if not articles:
        return []

    relevant_articles: list[Article] = []
    # Sort by summary length to group similar content types together.
    # Short-form sources (Nitter, Google News) and long-form sources (HN,
    # arXiv, blogs) get batched separately, preventing LLM from making
    # relative quality judgments across content types.
    sorted_articles = sorted(articles, key=lambda a: len(a.summary or ""))
    batches = [
        sorted_articles[i : i + MAX_ARTICLES_PER_BATCH]
        for i in range(0, len(sorted_articles), MAX_ARTICLES_PER_BATCH)
    ]
    total_batches = len(batches)

    for batch_idx, batch in enumerate(batches, 1):
        logger.info(
            "Filtering batch %d/%d (%d articles)...",
            batch_idx, total_batches, len(batch),
        )
        results = _call_deepseek(batch)
        if results is None:
            logger.warning("LLM filter failed for batch of %d articles, marking all as not relevant", len(batch))
            continue

        for article, result in zip(batch, results):
            article.relevant = bool(result.get("relevant", False))
            article.ai_summary = result.get("ai_summary", "")
            article.content_type = result.get("content_type", None)
            if article.relevant and article.content_type in ("article", "discussion"):
                relevant_articles.append(article)

    logger.info(
        "LLM filter: %d relevant+article/discussion out of %d articles",
        len(relevant_articles),
        len(articles),
    )
    return relevant_articles
