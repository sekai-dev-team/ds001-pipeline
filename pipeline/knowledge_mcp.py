"""Knowledge-MCP HTTP client for writing episodic notes.

Sends MCP JSON-RPC 2.0 requests to ``knowledge-mcp:8000/mcp``
to create episodic knowledge notes.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests

from pipeline.article import Article

logger = logging.getLogger(__name__)

KMCP_BASE_URL = "http://knowledge-mcp:8000/mcp"
REQUEST_TIMEOUT = 120  # notes now include full article text (~6KB+), needs longer timeout
MAX_RETRIES = 2  # retry up to 2 times on timeout
RETRY_DELAY = 5  # seconds between retries
_INITIALIZED = False


def _clean_nitter_html(text: str) -> str:
    """Strip HTML from Nitter RSS content and convert nitter.net links to x.com.

    * ``<br>`` / ``<br />`` → newline
    * ``<p>`` / ``</p>`` → removed
    * ``<a href="URL">text</a>`` → ``text (URL)``
    * ``<img …>`` → removed entirely
    * ``nitter.net`` → ``x.com`` in any remaining links
    * All other HTML tags are stripped.
    """
    if not text:
        return text
    # 1. Line breaks from <br>
    text = re.sub(r"<br\s*/?>", "\n", text)
    # 2. Remove <p> and </p> tags
    text = re.sub(r"</?p\s*/?>", "", text)
    # 3. Convert <a href="URL">text</a> → text (URL)
    text = re.sub(
        r'<a\s+href="([^"]*)"[^>]*>([^<]*)</a>',
        r"\2 (\1)",
        text,
    )
    # 4. Strip <img …> tags entirely (nitter.net/pic/ images are unrecoverable)
    text = re.sub(r"<img[^>]*>", "", text)
    # 5. Convert nitter.net → x.com in any remaining links
    text = re.sub(r"https?://nitter\.net/", "https://x.com/", text)
    # 6. Remove any remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    return text


def _make_note_content(article: Article) -> str:
    """Build the episodic note markdown body."""
    original_text = article.fulltext or article.summary
    cleaned_text = _clean_nitter_html(original_text or "")
    return (
        f"# {article.title}\n\n"
        f"**来源:** {article.source_name} | **摄入时间:** "
        f"{datetime.now(timezone.utc).isoformat()}\n\n"
        f"## 摘要\n{article.ai_summary}\n\n"
        f"## 原文\n{cleaned_text}\n"
    )


def _rpc_payload(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    }


def _initialize() -> bool:
    """Send MCP initialize request to the knowledge-MCP server.

    Must succeed before any ``tools/call`` requests are sent.
    Returns ``True`` on success, ``False`` on failure.
    """
    global _INITIALIZED  # noqa: PLW0603
    if _INITIALIZED:
        return True

    payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "ds001-pipeline",
                "version": "0.1.0",
            },
        },
    }

    try:
        resp = requests.post(
            KMCP_BASE_URL,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            logger.error("k-mcp initialize failed: %s", result["error"])
            return False
        _INITIALIZED = True
        logger.info("k-mcp initialize succeeded (protocol=%s)", result.get("protocolVersion", "unknown"))
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("k-mcp initialize request failed: %s", exc)
        return False


def write_note(article: Article) -> bool:
    """Write a single episodic note for *article* via the k-mcp API.

    Returns ``True`` on success, ``False`` on failure.
    """
    # Build a safe filename from the article title
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in article.title)
    safe_name = safe_name.strip().replace(" ", "_")[:120] or "untitled"
    filename = f"{safe_name}.md"

    content = _make_note_content(article)
    now_iso = datetime.now(timezone.utc).isoformat()

    payload = _rpc_payload("tools/call", {
        "name": "write_note",
        "arguments": {
            "path": filename,
            "content": content,
            "frontmatter": {
                "tags": ["ai-agent", "type/episodic", article.source_tag],
                "memory_type": "episodic",
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "ingested_at": now_iso,
                "source_url": article.url,
                "has_fulltext": article.has_fulltext,
            },
            "force": True,
        },
    })

    try:
        resp = requests.post(
            KMCP_BASE_URL,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()

        # Check for JSON-RPC error response
        if "error" in result:
            logger.error("k-mcp returned error for '%s': %s", article.title, result["error"])
            return False

        logger.info("Written note: %s (relevant=%s)", filename, article.relevant)
        return True

    except requests.exceptions.Timeout:
        logger.warning("k-mcp timeout for '%s' (attempt 1/%d)", article.title, MAX_RETRIES + 1)
        # Retry on timeout
        for attempt in range(1, MAX_RETRIES + 1):
            time.sleep(RETRY_DELAY * attempt)
            logger.info("Retrying write_note for '%s' (attempt %d/%d)", article.title, attempt + 1, MAX_RETRIES + 1)
            try:
                resp = requests.post(
                    KMCP_BASE_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                result = resp.json()
                if "error" not in result:
                    logger.info("Written note (after retry): %s", filename)
                    return True
                logger.error("k-mcp returned error for '%s' (retry %d): %s", article.title, attempt + 1, result["error"])
                return False
            except requests.exceptions.Timeout:
                logger.warning("k-mcp timeout for '%s' (attempt %d/%d)", article.title, attempt + 1, MAX_RETRIES + 1)
                if attempt == MAX_RETRIES:
                    logger.error("k-mcp timeout for '%s' after %d retries — giving up", article.title, MAX_RETRIES)
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("k-mcp request failed for '%s': %s", article.title, exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error writing note '%s': %s", article.title, exc)
        return False


def write_notes(articles: list[Article]) -> tuple[int, int]:
    """Write episodic notes for all relevant articles.

    Returns ``(success_count, total_count)``.
    """
    if not articles:
        return 0, 0

    # Initialize MCP session before any tool calls
    if not _initialize():
        logger.error("k-mcp initialize failed, skipping all write_note calls")
        return 0, len(articles)

    success = 0
    for article in articles:
        if write_note(article):
            success += 1
    return success, len(articles)
