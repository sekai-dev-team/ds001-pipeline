"""Knowledge-MCP HTTP client for writing episodic notes.

Sends MCP JSON-RPC 2.0 requests to ``knowledge-mcp:8000/mcp``
to create episodic knowledge notes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from pipeline.article import Article

logger = logging.getLogger(__name__)

KMCP_BASE_URL = "http://knowledge-mcp:8000/mcp"
REQUEST_TIMEOUT = 30
_INITIALIZED = False


def _make_note_content(article: Article) -> str:
    """Build the episodic note markdown body."""
    original_text = article.fulltext or article.summary
    return (
        f"# {article.title}\n\n"
        f"**来源:** {article.source_name} | **摄入时间:** "
        f"{datetime.now(timezone.utc).isoformat()}\n\n"
        f"## 摘要\n{article.ai_summary}\n\n"
        f"## 原文\n{original_text}\n"
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
        logger.error("k-mcp timeout for '%s'", article.title)
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
