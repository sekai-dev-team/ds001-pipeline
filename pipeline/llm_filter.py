"""LLM-based article filtering and summarization via DeepSeek API.

Uses DeepSeek V4 (deepseek-chat) with temperature 0.3.

Two-pass pipeline (v0.5):

**Pass 1** — Relevance classification (5 workers, ``max_tokens=256``).
  Uses only the RSS summary to determine relevance and content type.

  → Fulltext extraction for all Pass-1-passed articles.

**Pass 2** — Summarization (3 workers, ``max_tokens=1536``).
  Generates structured Chinese summaries using the first 4000 characters
  of fulltext. Falls back to RSS summary when fulltext is unavailable.
  Also extracts concept tags during the same LLM call.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from typing import Any

import requests

from pipeline.article import Article
from pipeline.fulltext import fetch_fulltext

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"  # Pass 1: fast/cheap relevance filter
DEEPSEEK_SUMMARIZE_MODEL = "deepseek-v4-pro"  # Pass 2: high-quality summarization

RELEVANCE_PROMPT = (
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
    'Return a JSON object with keys: relevant (bool), content_type (string).\n'
    "Always return valid JSON."
)

SUMMARIZE_PROMPT = (
    "You are an AI news summarizer and classifier. For the article below, "
    "generate a Chinese summary with these structured sections:\n"
    "**主题:** [1 sentence — what the article is about]\n"
    "**方法/发现:** [2-3 sentences — key method, result, or insight]\n"
    "**意义:** [1 sentence — why it matters]\n"
    "Total: at least 150 Chinese characters, at most 500.\n\n"
    "CONCEPT TAGS: Assign 0-3 concept tags to this article. "
    "Known tags that are high-frequency:\n{known_tags}\n\n"
    "Rules:\\n"
    "- Reuse an existing known tag name if the concept matches.\\n"
    "- Create a NEW tag (kebab-case English name) if the topic is new.\\n"
    "- Provide a one-sentence label_text for every tag.\\n"
    "- Keep proper nouns and technical terms in English (token, transformer, "
    "API, GPU, PyTorch, RLHF, etc.) — do NOT machine-translate them.\\n"
    "Format concept_tags as a JSON array: [{{\"name\": \"...\", \"label_text\": \"...\"}}]\n\n"
    'Return a JSON object with keys: ai_summary (string), concept_tags (array).\n'
    "Always return valid JSON."
)


def _call_deepseek(api_key: str, prompt: str, text: str, max_tokens: int, model: str = DEEPSEEK_MODEL) -> dict[str, Any] | None:
    """Shared DeepSeek API call helper.

    Sends *text* to DeepSeek with *prompt* as the system message and
    returns the parsed JSON response dict, or *None* on failure.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
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
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        return json.loads(cleaned)

    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException:
        return None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def _call_deepseek_relevance(article: Article) -> dict[str, Any] | None:
    """Pass 1: Quick relevance check using only the RSS summary.

    Returns a dict with ``relevant`` (bool) and ``content_type`` (string)
    keys, or *None* on failure.  ``max_tokens=256`` — just classification,
    no summary generation.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        return None

    text = (
        f"Title: {article.title}\n"
        f"Summary: {article.summary[:300]}\n"
        f"URL: {article.url}"
    )

    result = _call_deepseek(api_key, RELEVANCE_PROMPT, text, max_tokens=256)
    if result is None:
        logger.error("DeepSeek relevance check failed for: %s", article.title)
    return result


def _call_deepseek_summarize(
    api_key: str, article_text: str, known_tags: str = ""
) -> dict[str, Any] | None:
    """Pass 2: Full summarization using article fulltext + concept tag extraction.

    Uses the first 4000 characters of *article_text* (which should be either
    ``article.fulltext`` or ``article.summary``, prepared by the caller).

    Returns a dict with ``ai_summary`` (string) and ``concept_tags`` (list of
    dicts with ``name`` and ``label_text`` keys), or *None* on failure.
    ``max_tokens=1536`` — budget for structured Chinese summary + concept tags.
    """
    prompt = SUMMARIZE_PROMPT.format(known_tags=known_tags or "(no known tags yet)")

    result = _call_deepseek(api_key, prompt, article_text, max_tokens=1536, model=DEEPSEEK_SUMMARIZE_MODEL)
    if result is None:
        return None

    # Ensure concept_tags is present (default to empty list)
    if "concept_tags" not in result:
        result["concept_tags"] = []

    return result


def filter_articles(
    articles: list[Article],
    api_key: str,
    tag_library: dict[str, dict] | None = None,
) -> list[Article]:
    """Two-pass filter: relevance check → fulltext extraction → summarization.

    **Pass 1** (5 workers via ``ThreadPoolExecutor``):
    Quick relevance classification using only the RSS summary
    (``_call_deepseek_relevance``, ``max_tokens=256``).

    **Fulltext extraction** (sequential, via :func:`pipeline.fulltext.fetch_fulltext`):
    Fetch full HTML content for every article that passed Pass 1.

    **Pass 2** (3 workers via ``ThreadPoolExecutor``):
    Generate structured Chinese summaries using the first 4000 characters of
    fulltext (``_call_deepseek_summarize``, ``max_tokens=1536``).  Falls back
    to the RSS summary for articles where fulltext extraction failed.
    Also extracts concept tags during the same LLM call.

    Errors are isolated per article — one failure does not affect others.
    Each article is updated in-place with ``relevant``, ``ai_summary``,
    ``concept_tags``, ``content_type``, ``fulltext``, and ``has_fulltext``.
    Only articles deemed relevant AND classified as "article" or "discussion"
    content type are returned.  ``link_only`` and ``low_quality`` are rejected.
    """
    if not articles:
        return []

    # ------------------------------------------------------------------
    # Pass 1: Relevance check (5 workers, RSS summary only)
    # ------------------------------------------------------------------
    pass1_passed: list[Article] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_article: dict[concurrent.futures.Future, Article] = {
            executor.submit(_call_deepseek_relevance, article): article
            for article in articles
        }

        for future in concurrent.futures.as_completed(future_to_article):
            article = future_to_article[future]
            try:
                result = future.result()
                if result is None:
                    logger.info(
                        "Pass 1: no result for '%s', marking not relevant",
                        article.title,
                    )
                    continue

                article.relevant = bool(result.get("relevant", False))
                article.content_type = result.get("content_type", None)

                if article.relevant and article.content_type in ("article", "discussion"):
                    pass1_passed.append(article)

            except Exception as exc:
                logger.error(
                    "Pass 1 error for '%s': %s", article.title, exc,
                )

    logger.info(
        "Pass 1: %d relevant+article/discussion out of %d articles",
        len(pass1_passed), len(articles),
    )

    if not pass1_passed:
        return []

    # ------------------------------------------------------------------
    # Fulltext extraction for Pass 1 results
    # ------------------------------------------------------------------
    fulltext_success = 0
    for article in pass1_passed:
        result = fetch_fulltext(article.url)
        if result is not None:
            article.fulltext = result
            article.has_fulltext = True
            fulltext_success += 1
        else:
            article.has_fulltext = False

    logger.info(
        "Fulltext: %d/%d extracted for relevant articles",
        fulltext_success, len(pass1_passed),
    )

    # ------------------------------------------------------------------
    # Pass 2: Summarization + concept tag extraction (3 workers, fulltext first 4000 chars)
    # ------------------------------------------------------------------
    # Build known_tags string from tag library (only high-frequency: article_count > 5)
    known = ""
    if tag_library:
        high_freq = {k: v for k, v in tag_library.items() if v.get("article_count", 0) > 5}
        if high_freq:
            known = "\n".join(
                f'- {k}: "{v.get("label_text", "")}"' for k, v in sorted(high_freq.items())
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_article: dict[concurrent.futures.Future, Article] = {}

        for article in pass1_passed:
            # Prepare article text (prefer fulltext, fall back to RSS summary)
            if article.has_fulltext and article.fulltext:
                source_text = article.fulltext
            else:
                source_text = article.summary[:300]

            article_text = (
                f"Title: {article.title}\n"
                f"Content: {source_text}\n"
                f"URL: {article.url}"
            )

            future = executor.submit(
                _call_deepseek_summarize, api_key, article_text, known_tags=known
            )
            future_to_article[future] = article

        for future in concurrent.futures.as_completed(future_to_article):
            article = future_to_article[future]
            try:
                result = future.result()
                if result is not None:
                    article.ai_summary = result.get("ai_summary", "")
                    article.concept_tags = result.get("concept_tags", [])
            except Exception as exc:
                logger.error(
                    "Pass 2 error for '%s': %s", article.title, exc,
                )

    logger.info(
        "Pass 2: summarization complete for %d articles",
        len(pass1_passed),
    )

    return pass1_passed
