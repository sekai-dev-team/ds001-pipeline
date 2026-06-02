"""Daily digest generation via DeepSeek API.

Produces a markdown daily digest with:
- 3-5 key signals/themes extracted across articles (not per-article summaries)
- A full article index with title, source, and 1-sentence summary

References ``pipeline/llm_filter.py`` for DeepSeek API patterns.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import requests

from pipeline.article import Article

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_ARTICLES_PER_BATCH = 25

SIGNAL_SYSTEM_PROMPT = (
    "你是一名AI研究分析师，负责生成每日情报摘要。"
    "以下是一批关于AI架构、大语言模型、AI智能体、"
    "开源AI以及AI工程范式的文章。\n\n"
    "请从这些文章中提炼出3-5个跨文章的关键信号或主题。"
    "不要逐一总结每篇文章，而是要识别跨领域的模式、"
    "新兴趋势或重要进展。\n\n"
    '返回一个JSON数组，每个对象包含以下字段：\n'
    '  - "name": 简短的信号名称（2-6个汉字，简洁明了）\n'
    '  - "why": 1-2句话解释为什么这个信号值得关注（使用中文）\n'
    '  - "articles": 与该信号相关的文章标题列表（精确匹配英文原标题）\n\n'
    "仅返回有效的JSON，不要添加额外文字，不要使用Markdown代码块。"
)


def _date_from_timestamp(timestamp: str) -> str:
    """Extract a human-readable date (YYYY-MM-DD) from an ISO timestamp."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        # Fallback: use today's date
        return datetime.now().strftime("%Y-%m-%d")


def _call_deepseek_for_signals(articles: list[Article]) -> list[dict[str, Any]] | None:
    """Send a batch of articles to DeepSeek for cross-article signal extraction.

    Returns a list of signal dicts (``name``, ``why``, ``articles``), or
    *None* on failure.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set")
        return None

    article_texts = []
    for i, a in enumerate(articles, 1):
        article_texts.append(
            f"[{i}] Title: {a.title}\n    Source: {a.source_name}\n"
            f"    Summary: {(a.ai_summary or a.summary or '')[:300]}"
        )

    user_content = (
        "请分析以下文章，提炼出3-5个关键信号：\n\n"
        + "\n---\n".join(article_texts)
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SIGNAL_SYSTEM_PROMPT},
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
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        results: list[dict[str, Any]] = json.loads(cleaned)
        # Validate that each result has required keys
        validated = []
        for r in results:
            if isinstance(r, dict) and "name" in r and "why" in r:
                validated.append(r)
        if not validated:
            logger.warning("No valid signals returned from DeepSeek")
            return None
        return validated

    except requests.exceptions.Timeout:
        logger.error("DeepSeek API timeout after 120s")
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("DeepSeek API request failed: %s", exc)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Failed to parse DeepSeek response: %s", exc)
        return None


def _render_signals(signals: list[dict[str, Any]]) -> str:
    """Render the signals section of the digest markdown."""
    lines = ["## 🔍 今日信号"]
    for idx, signal in enumerate(signals, 1):
        name = signal.get("name", f"Signal {idx}")
        why = signal.get("why", "")
        articles_list = signal.get("articles", [])
        articles_str = ", ".join(articles_list) if articles_list else ""
        lines.append(f"### 信号 {idx}: {name}")
        lines.append(why)
        if articles_str:
            lines.append(f"> 来源: {articles_str}")
        lines.append("")  # blank line separator
    return "\n".join(lines)


def _render_source_stats(per_source: dict[str, dict[str, int]]) -> str:
    """Render a per-source statistics table as markdown."""
    lines = ["## 📊 来源统计", ""]
    lines.append("| 来源 | 抓取 | 相关 | 入库 |")
    lines.append("|------|------|------|------|")
    for source_name, counts in per_source.items():
        fetched = counts.get("fetched", 0)
        relevant = counts.get("relevant", 0)
        ingested = counts.get("ingested", 0)
        lines.append(f"| {source_name} | {fetched} | {relevant} | {ingested} |")
    return "\n".join(lines)


def _render_article_list(articles: list[Article]) -> str:
    """Render the article index section of the digest markdown."""
    lines = [f"## 📋 摄入文章 ({len(articles)}篇)"]
    for idx, article in enumerate(articles, 1):
        summary = (article.ai_summary or article.summary or "No summary available")
        # Truncate long summaries for digest readability
        if len(summary) > 200:
            summary = summary[:197] + "..."
        lines.append(f"{idx}. **{article.title}** [{article.source_name}]")
        lines.append(f"   {summary}")
    return "\n".join(lines)


def generate_digest(
    articles: list[Article],
    mode: str,
    timestamp: str,
    per_source: dict[str, dict[str, int]] | None = None,
) -> str:
    """Generate a daily digest markdown document.

    Parameters
    ----------
    articles:
        The list of quality-passed, ingested articles to digest.
    mode:
        Pipeline mode (``"daily"`` or ``"hn-arxiv"``) — included as metadata.
    timestamp:
        ISO-8601 pipeline timestamp.
    per_source:
        Optional per-source statistics dict mapping source names to counts
        (``fetched``, ``relevant``, ``ingested``). Rendered as a markdown
        table between the signals and article list sections.

    Returns
    -------
    str
        Complete markdown digest ready for vault storage.
    """
    if not articles:
        logger.warning("generate_digest called with empty article list")
        date_str = _date_from_timestamp(timestamp)
        return (
            f"# 📰 Yui 日报 · {date_str}\n\n"
            f"*Pipeline mode: {mode}*\n\n"
            "No articles to digest.\n"
        )

    date_str = _date_from_timestamp(timestamp)

    # Header
    header = f"# 📰 Yui 日报 · {date_str}\n\n"

    # --- Signals section (via LLM) ---
    signals: list[dict[str, Any]] = []
    batches = [
        articles[i : i + MAX_ARTICLES_PER_BATCH]
        for i in range(0, len(articles), MAX_ARTICLES_PER_BATCH)
    ]

    for batch_idx, batch in enumerate(batches, 1):
        logger.info(
            "Extracting signals for batch %d/%d (%d articles)...",
            batch_idx, len(batches), len(batch),
        )
        batch_signals = _call_deepseek_for_signals(batch)
        if batch_signals:
            signals.extend(batch_signals)
        else:
            logger.warning(
                "Signal extraction failed for batch %d/%d",
                batch_idx, len(batches),
            )

    # Deduplicate signals by name (case-insensitive)
    if signals:
        seen_names: set[str] = set()
        unique_signals: list[dict[str, Any]] = []
        for s in signals:
            name_lower = s.get("name", "").lower().strip()
            if name_lower and name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_signals.append(s)
        signals = unique_signals

    # Render sections
    if signals:
        body = _render_signals(signals)
    else:
        logger.info("No signals extracted — falling back to article-list-only digest")
        body = "## 🔍 今日信号\n\n*信号分析暂不可用 — 以下为完整文章索引。*\n\n"

    # Per-source statistics (if available)
    if per_source:
        body += "\n" + _render_source_stats(per_source)

    body += "\n" + _render_article_list(articles)

    return header + body
