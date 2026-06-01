"""Knowledge-MCP HTTP client for writing episodic notes.

Sends MCP JSON-RPC 2.0 requests to ``knowledge-mcp:8000/mcp``
to create episodic knowledge notes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests
import yaml

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


# ------------------------------------------------------------------
# k-mcp Embed Client
# ------------------------------------------------------------------


def _kmcp_embed(text: str) -> list[float] | None:
    """Call k-mcp embed tool and return embedding vector.

    Returns ``None`` if the k-mcp server is unreachable or returns an error.
    """
    if not _initialize():
        return None
    payload = _rpc_payload("tools/call", {
        "name": "embed",
        "arguments": {"text": text},
    })
    try:
        resp = requests.post(
            KMCP_BASE_URL,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            logger.warning("k-mcp embed returned error: %s", result["error"])
            return None
        content = result.get("result", {}).get("content", [])
        if content and isinstance(content, list):
            text_content = content[0].get("text", "[]")
            return json.loads(text_content)
    except Exception as exc:
        logger.warning("k-mcp embed request failed: %s", exc)
    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        logger.warning(
            "Embedding dimension mismatch: %d vs %d — treating as cos=0.0",
            len(a), len(b),
        )
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ------------------------------------------------------------------
# Tag Library File I/O
# ------------------------------------------------------------------


def _parse_frontmatter(content: str) -> dict[str, Any] | None:
    """Parse YAML frontmatter from a markdown string.

    Returns the parsed dict, or ``None`` if no frontmatter is found.
    """
    if not content.startswith("---"):
        return None
    end = content.find("---", 3)
    if end == -1:
        return None
    yaml_str = content[3:end].strip()
    if not yaml_str:
        return None
    try:
        return yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        return None


def _read_tag_library() -> dict[str, dict]:
    """Read all .md files in /vault/tags/, parse frontmatter.

    Returns a dict mapping tag name → tag info dict.
    """
    tags_dir = os.path.join(VAULT_PATH, "tags")
    library: dict[str, dict] = {}
    if not os.path.isdir(tags_dir):
        return library
    for fname in sorted(os.listdir(tags_dir)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(tags_dir, fname)
        try:
            with open(fpath, "r") as f:
                content = f.read()
        except OSError:
            continue
        fm = _parse_frontmatter(content)
        if not fm:
            continue
        name = fm.get("name") or fname[:-3]
        emb_raw = fm.get("embedding")
        if isinstance(emb_raw, str):
            try:
                emb_raw = json.loads(emb_raw)
            except (json.JSONDecodeError, TypeError):
                emb_raw = []
        library[name] = {
            "name": name,
            "embedding": emb_raw if isinstance(emb_raw, list) else [],
            "label_text": fm.get("label_text", ""),
            "article_count": fm.get("article_count", 0),
            "concept_page": fm.get("concept_page"),
            "first_seen": fm.get("first_seen"),
            "last_seen": fm.get("last_seen"),
        }
    return library


def _create_tag_note(name: str, label_text: str, embedding: list[float]) -> None:
    """Create /vault/tags/{name}.md with frontmatter and body."""
    tags_dir = os.path.join(VAULT_PATH, "tags")
    os.makedirs(tags_dir, exist_ok=True)
    emb_str = json.dumps(embedding)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = f"""---
name: {name}
embedding: {emb_str}
label_text: "{label_text}"
article_count: 1
concept_page:
first_seen: {date_str}
last_seen: {date_str}
---

# {name}

{label_text}
"""
    filepath = os.path.join(tags_dir, f"{name}.md")
    with open(filepath, "w") as f:
        f.write(content)
    logger.info("Created tag note: %s", filepath)


def _update_tag_note(name: str, updates: dict) -> None:
    """Patch frontmatter of /vault/tags/{name}.md.

    Reads the existing note, applies *updates* to the frontmatter dict,
    and rewrites.  If the file does not exist, this is a no-op.
    """
    fpath = os.path.join(VAULT_PATH, "tags", f"{name}.md")
    if not os.path.isfile(fpath):
        logger.warning("_update_tag_note: tag file not found: %s", fpath)
        return

    try:
        with open(fpath, "r") as f:
            content = f.read()
    except OSError:
        logger.warning("_update_tag_note: cannot read tag file: %s", fpath)
        return

    fm = _parse_frontmatter(content)
    if not fm:
        logger.warning("_update_tag_note: no frontmatter in tag file: %s", fpath)
        return

    # Apply updates
    fm.update(updates)

    # Rebuild the file
    body_start = content.find("---", 3)
    body = ""
    if body_start != -1:
        body = content[body_start + 3:].lstrip("\n")

    new_fm_yaml = yaml.dump(
        fm, allow_unicode=True, default_flow_style=None, sort_keys=False
    ).strip()
    new_content = f"---\n{new_fm_yaml}\n---\n\n{body}"

    try:
        with open(fpath, "w") as f:
            f.write(new_content)
    except OSError as exc:
        logger.error("_update_tag_note: cannot write tag file: %s: %s", fpath, exc)


# ------------------------------------------------------------------
# Tag Processing (post-summarization)
# ------------------------------------------------------------------


def process_tags(articles: list[Article]) -> None:
    """For each article, process concept_tags: dedup against tag library,
    create new tags if needed, and set article.topic_tags for frontmatter.

    Modifies articles in-place: sets ``article.topic_tags`` (list of tag names).
    """
    tag_library = _read_tag_library()
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for article in articles:
        article.topic_tags = []
        if not hasattr(article, 'concept_tags') or not article.concept_tags:
            continue

        for tag in article.concept_tags:
            name = tag.get("name", "").strip().lower().replace(" ", "-")
            label = tag.get("label_text", "").strip()
            if not name or not label:
                continue

            if name in tag_library:
                # Existing tag: reuse
                article.topic_tags.append(name)
                tag_library[name]["article_count"] = tag_library[name].get("article_count", 0) + 1
                _update_tag_note(name, {
                    "article_count": tag_library[name]["article_count"],
                    "last_seen": date_str,
                })
            else:
                # New tag: embed and dedup
                embedding = _kmcp_embed(label)
                if not embedding:
                    # Fallback: create the tag without embedding
                    _create_tag_note(name, label, [])
                    tag_library[name] = {
                        "name": name, "embedding": [], "label_text": label,
                        "article_count": 1, "concept_page": None,
                        "first_seen": date_str, "last_seen": date_str,
                    }
                    article.topic_tags.append(name)
                    continue

                # Cosine dedup against existing library tags
                best_match = None
                best_score = 0.0
                for lib_name, lib_info in tag_library.items():
                    lib_emb = lib_info.get("embedding")
                    if not lib_emb:
                        continue
                    score = _cosine_similarity(embedding, lib_emb)
                    if score > best_score:
                        best_score = score
                        best_match = lib_name

                if best_score > 0.90 and best_match:
                    # Semantic duplicate: reuse existing tag
                    article.topic_tags.append(best_match)
                    tag_library[best_match]["article_count"] = (
                        tag_library[best_match].get("article_count", 0) + 1
                    )
                    _update_tag_note(best_match, {
                        "article_count": tag_library[best_match]["article_count"],
                        "last_seen": date_str,
                    })
                else:
                    # Genuinely new tag
                    _create_tag_note(name, label, embedding)
                    tag_library[name] = {
                        "name": name, "embedding": embedding, "label_text": label,
                        "article_count": 1, "concept_page": None,
                        "first_seen": date_str, "last_seen": date_str,
                    }
                    article.topic_tags.append(name)

    logger.info(
        "process_tags: processed %d articles, tag library has %d tags",
        len(articles), len(tag_library),
    )


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

    # Build tags list with topic/ prefix for concept tags
    tags = ["ai-agent", "type/episodic", article.source_tag]
    for tag_name in getattr(article, 'topic_tags', []):
        tags.append(f"topic/{tag_name}")

    payload = _rpc_payload("tools/call", {
        "name": "write_note",
        "arguments": {
            "path": filename,
            "content": content,
            "frontmatter": {
                "tags": tags,
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


# -- Filesystem-first bulk write (avoids per-note embedding model calls) --

VAULT_PATH = os.environ.get("DS001_VAULT_PATH", "/vault")


def write_notes_fs(articles: list[Article]) -> tuple[int, int]:
    """Write article .md files directly to the vault filesystem.

    Each article is formatted as markdown with YAML frontmatter and written
    to ``DS001_VAULT_PATH`` (default ``/vault``).  No HTTP calls are made
    per article — this avoids triggering ``incremental_index()`` (and the
    embedding model) for every write.  Call ``reindex_vault()`` **once**
    after all files are written to bulk-reindex the vault.

    Returns ``(success_count, total_count)``.
    """
    if not articles:
        return 0, 0

    os.makedirs(VAULT_PATH, exist_ok=True)
    success = 0

    for article in articles:
        safe_name = "".join(
            c if c.isalnum() or c in " -_" else "_" for c in article.title
        )
        safe_name = safe_name.strip().replace(" ", "_")[:120] or "untitled"
        filename = f"{safe_name}.md"
        filepath = os.path.join(VAULT_PATH, filename)

        content = _make_note_content(article)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Build tags list with topic/ prefix for concept tags
        tags = ["ai-agent", "type/episodic", article.source_tag]
        for tag_name in getattr(article, 'topic_tags', []):
            tags.append(f"topic/{tag_name}")

        frontmatter = {
            "tags": tags,
            "memory_type": "episodic",
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "ingested_at": now_iso,
            "source_url": article.url,
            "has_fulltext": article.has_fulltext,
        }

        try:
            fm_yaml = yaml.dump(
                frontmatter, allow_unicode=True, default_flow_style=None, sort_keys=False
            ).strip()
            body = f"---\n{fm_yaml}\n---\n\n{content}"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(body)
            logger.info("Written note to filesystem: %s (relevant=%s)", filename, article.relevant)
            success += 1
        except OSError as exc:
            logger.error("Failed to write note '%s' to filesystem: %s", filename, exc)

    logger.info(
        "Filesystem write complete: %d/%d notes written to %s",
        success, len(articles), VAULT_PATH,
    )
    return success, len(articles)


def reindex_vault() -> bool:
    """Send a single ``reindex`` RPC call to k-mcp to bulk-reindex all notes.

    Call this **once** after ``write_notes_fs()`` finishes writing all files.
    Uses a longer timeout (360 s) because a full rebuild touches every .md
    file in the vault and may invoke the embedding model many times in
    sequence (but crucially, within a single process that does not stack
    incremental-index overhead).

    Returns ``True`` on success, ``False`` on failure.
    """
    if not _initialize():
        logger.error("k-mcp initialize failed, cannot reindex vault")
        return False

    payload = _rpc_payload("tools/call", {
        "name": "reindex",
        "arguments": {},
    })

    try:
        resp = requests.post(
            KMCP_BASE_URL,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=360,  # bulk reindex can take minutes with embedding model
        )
        resp.raise_for_status()
        result = resp.json()

        if "error" in result:
            logger.error("k-mcp reindex returned error: %s", result["error"])
            return False

        logger.info("Vault reindex succeeded: %s", result.get("result", "ok"))
        return True

    except requests.exceptions.Timeout:
        logger.error("k-mcp reindex timed out after 360s")
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("k-mcp reindex request failed: %s", exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error during reindex: %s", exc)
        return False


def write_digest(digest_md: str, timestamp: str) -> bool:
    """Write the daily digest markdown document to the knowledge-mcp vault.

    Parameters
    ----------
    digest_md:
        The complete markdown digest content.
    timestamp:
        ISO-8601 pipeline timestamp (used to derive the date for the filename).

    Returns
    -------
    bool
        ``True`` if the digest was successfully saved.
    """
    # Derive date from timestamp
    try:
        dt = datetime.fromisoformat(timestamp)
        date_str = dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    filename = f"daily-digest/{date_str}.md"

    # Initialize MCP session before tool call
    if not _initialize():
        logger.error("k-mcp initialize failed, cannot write digest")
        return False

    frontmatter = {
        "tags": ["ai-agent", "type/digest", "daily-digest"],
        "memory_type": "episodic",
        "date": date_str,
        "ingested_at": timestamp,
    }

    payload = _rpc_payload("tools/call", {
        "name": "write_note",
        "arguments": {
            "path": filename,
            "content": digest_md,
            "frontmatter": frontmatter,
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

        if "error" in result:
            logger.error("k-mcp returned error for digest '%s': %s", filename, result["error"])
            return False

        logger.info("Written digest: %s", filename)
        return True

    except requests.exceptions.Timeout:
        logger.warning("k-mcp timeout for digest '%s', retrying...", filename)
        for attempt in range(1, MAX_RETRIES + 1):
            import time as _time
            _time.sleep(RETRY_DELAY * attempt)
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
                    logger.info("Written digest (after retry): %s", filename)
                    return True
                logger.error("k-mcp returned error for digest (retry %d): %s", attempt + 1, result["error"])
                return False
            except requests.exceptions.Timeout:
                logger.warning("k-mcp timeout for digest (attempt %d/%d)", attempt + 1, MAX_RETRIES + 1)
                if attempt == MAX_RETRIES:
                    logger.error("k-mcp timeout for digest after %d retries — giving up", MAX_RETRIES)
        return False
    except requests.exceptions.RequestException as exc:
        logger.error("k-mcp request failed for digest: %s", exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error writing digest: %s", exc)
        return False
