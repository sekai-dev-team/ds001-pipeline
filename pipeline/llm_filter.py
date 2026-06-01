"""LLM-based article filtering and summarization via DeepSeek API.

Uses DeepSeek V4 (deepseek-chat) with temperature 0.3 to determine
relevance and generate Chinese summaries.

Each article is processed in its own API call (no batching) with up to
5 concurrent workers via ThreadPoolExecutor.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from typing import Any

import requests

from pipeline.article import Article

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = (
    "You are an AI news filter. For the article below:\n"
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
    'Return a JSON object with keys: relevant (bool), ai_summary (string), '
    'content_type (string).\n'
    "Always return valid JSON."
)


def _call_deepseek_single(article: Article) -> dict[str, Any] | None:
    """Send a single article to DeepSeek for filtering.

    Returns a dict with ``relevant``, ``ai_summary``, and
    ``content_type`` keys, or *None* on failure.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        return None

    article_text = (
        f"Title: {article.title}\n"
        f"Summary: {article.summary[:300]}\n"
        f"URL: {article.url}"
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article_text},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
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

        result: dict[str, Any] = json.loads(cleaned)
        return result

    except requests.exceptions.Timeout:
        logger.error("DeepSeek API timeout for article: %s", article.title)
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("DeepSeek API request failed for '%s': %s", article.title, exc)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Failed to parse DeepSeek response for '%s': %s", article.title, exc)
        return None


def filter_articles(articles: list[Article]) -> list[Article]:
    """Filter and summarize articles using DeepSeek API.

    Each article is processed in its own API call (no batching) with up to
    5 concurrent workers via ``ThreadPoolExecutor``.
    Errors are isolated per article — one failure does not affect others.
    Each article is updated in-place with ``relevant``, ``ai_summary``,
    and ``content_type``.
    Only articles deemed relevant AND classified as "article" or "discussion"
    content type are returned. ``link_only`` and ``low_quality`` are rejected.
    """
    if not articles:
        return []

    relevant_articles: list[Article] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_article: dict[concurrent.futures.Future, Article] = {
            executor.submit(_call_deepseek_single, article): article
            for article in articles
        }

        for future in concurrent.futures.as_completed(future_to_article):
            article = future_to_article[future]
            try:
                result = future.result()
                if result is None:
                    logger.info(
                        "LLM filter: no result for '%s', marking not relevant",
                        article.title,
                    )
                    continue

                article.relevant = bool(result.get("relevant", False))
                article.ai_summary = result.get("ai_summary", "")
                article.content_type = result.get("content_type", None)

                if article.relevant and article.content_type in ("article", "discussion"):
                    relevant_articles.append(article)

            except Exception as exc:
                logger.error(
                    "LLM filter error for article '%s': %s",
                    article.title, exc,
                )

    logger.info(
        "LLM filter: %d relevant+article/discussion out of %d articles",
        len(relevant_articles),
        len(articles),
    )
    return relevant_articles
