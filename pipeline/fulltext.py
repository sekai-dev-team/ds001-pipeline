"""Full-text extraction from article URLs.

Primary: trafilatura (extract markdown from HTML).
Fallback: readability-lxml + html2text.
arXiv: dedicated API for paper abstracts.
HN: Firebase API for discussion threads.
Reddit: .json API for discussion threads.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL pattern classification
# ---------------------------------------------------------------------------


def url_pattern_hint(url: str) -> str:
    """Classify a URL into a content-type hint for extraction strategy.

    Returns one of: ``"canonical_article"``, ``"hn_discussion"``,
    ``"reddit_discussion"``, ``"discussion_site"``, ``"arxiv_paper"``,
    ``"tweet"``.
    """
    if not url:
        return "canonical_article"

    # arXiv abstract pages -> dedicated API
    if re.search(r"arxiv\.org/abs/\d+", url):
        return "arxiv_paper"

    # Nitter tweets (JS-rendered pages)
    if re.search(r"nitter\.net/\w+/status/", url):
        return "tweet"

    # Discussion sites
    if "news.ycombinator.com/item" in url:
        return "hn_discussion"
    if "reddit.com" in url:
        return "reddit_discussion"
    if ".mastodon." in url or "mastodon.social" in url:
        return "discussion_site"

    return "canonical_article"


# ---------------------------------------------------------------------------
# Discussion site extraction
# ---------------------------------------------------------------------------


def _fetch_hn_discussion(url: str) -> Optional[str]:
    """Fetch HN discussion via Firebase API (public, no auth).

    Extracts the item ID from a ``news.ycombinator.com/item?id=XXXXX``
    URL, fetches the post metadata via the Firebase API, then fetches
    the top-level comments (up to 30) individually.

    Returns markdown-formatted content or *None* on failure.
    """
    match = re.search(r"news\.ycombinator\.com/item\?id=(\d+)", url)
    if not match:
        return None

    item_id = match.group(1)
    api_url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"

    try:
        import requests as req_lib

        resp = req_lib.get(api_url, timeout=30)
        resp.raise_for_status()
        item = resp.json()
        if not item:
            return None

        title = item.get("title", "HN Discussion")
        text = item.get("text", "")
        kids = item.get("kids", [])

        lines = [f"# {title}"]
        if text:
            lines.append("")
            lines.append(text)

        if kids:
            lines.append("")
            lines.append("## Top Comments")
            for cid in kids[:30]:
                try:
                    cresp = req_lib.get(
                        f"https://hacker-news.firebaseio.com/v0/item/{cid}.json",
                        timeout=15,
                    )
                    cresp.raise_for_status()
                    comment = cresp.json()
                    if comment and comment.get("text"):
                        author = comment.get("by", "anonymous")
                        lines.append(f"**{author}:** {comment['text']}\n")
                except Exception:
                    logger.debug("Failed to fetch HN comment %s", cid)
                    continue

        result = "\n".join(lines).strip()
        return result if result else None

    except Exception:
        logger.exception("HN API request failed for: %s", url)
        return None


def _fetch_reddit_discussion(url: str) -> Optional[str]:
    """Fetch Reddit discussion via the public .json API.

    Appends ``.json`` to the Reddit URL, parses the post metadata and
    threaded comments (up to 3 levels deep, top 20 comments).

    Requires a ``User-Agent`` header (Reddit policy).

    Returns markdown-formatted content or *None* on failure.
    """
    # Normalise: strip trailing slash, append .json
    json_url = url.rstrip("/") + ".json"

    try:
        import requests as req_lib

        def _try_fetch(target_url: str) -> Optional[list]:
            resp = req_lib.get(
                target_url,
                timeout=30,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; DS-001-Pipeline/0.2.3)"
                    ),
                },
            )
            if resp.status_code == 403:
                return None
            resp.raise_for_status()
            return resp.json()

        data = _try_fetch(json_url)
        if data is None:
            # Fallback: try www.reddit.com when old.reddit.com returns 403
            fallback_url = json_url.replace("old.reddit.com", "www.reddit.com")
            if fallback_url != json_url:
                logger.info(
                    "old.reddit.com returned 403, falling back to %s",
                    fallback_url,
                )
                data = _try_fetch(fallback_url)

        if data is None:
            return None

        if not isinstance(data, list) or len(data) < 2:
            return None

        post_data = data[0]["data"]["children"][0]["data"]
        title = post_data.get("title", "Reddit Discussion")
        selftext = post_data.get("selftext", "")

        lines = [f"# {title}"]
        if selftext:
            lines.append("")
            lines.append(selftext)

        comments_data = data[1]["data"]["children"]

        def _walk_reddit_comments(
            children: list,
            depth: int = 0,
            max_depth: int = 3,
            max_comments: int = 20,
        ) -> list[str]:
            """Recursively format Reddit comments as markdown."""
            result: list[str] = []
            count = 0
            for child in children:
                if count >= max_comments:
                    break
                if child.get("kind") != "t1":
                    continue
                cdata = child.get("data", {})
                author = cdata.get("author", "[deleted]")
                body = cdata.get("body", "")
                if not body:
                    continue
                indent = "  " * depth
                result.append(f"{indent}**{author}:** {body}\n")
                count += 1

                replies = cdata.get("replies", {})
                if depth < max_depth and isinstance(replies, dict):
                    reply_children = (
                        replies.get("data", {}).get("children", [])
                    )
                    result.extend(
                        _walk_reddit_comments(
                            reply_children,
                            depth + 1,
                            max_depth,
                            max_comments - count,
                        )
                    )
                    # Re-count after recursion
                    count = sum(
                        1
                        for r in result
                        if r.lstrip().startswith("**")
                    )

            return result

        formatted = _walk_reddit_comments(comments_data)
        if formatted:
            lines.append("")
            lines.append("## Comments")
            lines.extend(formatted)

        result = "\n".join(lines).strip()
        return result if result else None

    except Exception:
        logger.exception("Reddit API request failed for: %s", url)
        return None


# ---------------------------------------------------------------------------
# arXiv API extraction
# ---------------------------------------------------------------------------


def _fetch_arxiv_abstract(url: str) -> Optional[str]:
    """Fetch paper abstract via the arXiv API for an ``/abs/XXXX`` URL.

    Returns markdown-formatted ``# Title`` + abstract text, or *None*
    on failure.
    """
    match = re.search(r"arxiv\.org/abs/(\d+(?:\.\d+)?)", url)
    if not match:
        return None

    arxiv_id = match.group(1)
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    try:
        import requests as req_lib

        resp = req_lib.get(api_url, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        title_el = root.find(".//atom:title", ns)
        summary_el = root.find(".//atom:summary", ns)

        title = (
            title_el.text.strip()
            if title_el is not None and title_el.text
            else "Untitled"
        )
        summary = (
            summary_el.text.strip()
            if summary_el is not None and summary_el.text
            else ""
        )

        if summary:
            return f"# {title}\n\n{summary}"
        return None
    except ET.ParseError:
        logger.exception("arXiv API XML parse failed for: %s", url)
        return None
    except Exception:
        logger.exception("arXiv API request failed for: %s", url)
        return None


# ---------------------------------------------------------------------------
# Primary: trafilatura
# ---------------------------------------------------------------------------

TRAFILATURA_AVAILABLE = False
try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    pass


def _extract_trafilatura(url: str, timeout: int = 30) -> Optional[str]:
    """Extract fulltext via trafilatura (markdown output).

    Uses ``requests.get()`` for the HTTP fetch (so we can control the
    timeout), then passes the response text to ``trafilatura.extract()``.

    Returns *None* if extraction fails or trafilatura is not installed.
    """
    if not TRAFILATURA_AVAILABLE:
        return None

    try:
        import requests as req_lib

        resp = req_lib.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; DS-001-Pipeline/0.2; "
                    "+https://github.com/sekai-dev-team/ds001-pipeline)"
                ),
            },
        )
        resp.raise_for_status()
        result = trafilatura.extract(resp.text, output_format="markdown")
        if result is None:
            logger.debug("trafilatura.extract returned None for: %s", url)
            return None
        cleaned = result.strip()
        return cleaned if cleaned else None
    except Exception:
        logger.exception("trafilatura extraction failed for: %s", url)
        return None


# ---------------------------------------------------------------------------
# Fallback: readability-lxml + html2text
# ---------------------------------------------------------------------------

READABILITY_AVAILABLE = False
HTML2TEXT_AVAILABLE = False
try:
    from readability import Document as ReadabilityDoc

    READABILITY_AVAILABLE = True
except ImportError:
    pass

try:
    import html2text

    HTML2TEXT_AVAILABLE = True
except ImportError:
    pass


def _extract_readability(url: str, timeout: int = 30) -> Optional[str]:
    """Extract fulltext via readability-lxml + html2text.

    Returns *None* if extraction fails or dependencies are not installed.
    """
    if not READABILITY_AVAILABLE or not HTML2TEXT_AVAILABLE:
        return None

    try:
        import requests

        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; DS-001-Pipeline/0.2; "
                    "+https://github.com/sekai-dev-team/ds001-pipeline)"
                )
            },
        )
        resp.raise_for_status()

        doc = ReadabilityDoc(resp.text)
        body_html = doc.summary()

        converter = html2text.HTML2Text()
        converter.body_width = 0  # no line wrapping
        converter.ignore_links = False
        converter.ignore_images = False

        markdown = converter.handle(body_html)
        cleaned = markdown.strip()
        return cleaned if cleaned else None
    except Exception:
        logger.exception("readability-lxml extraction failed for: %s", url)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_fulltext(url: str, timeout: int = 30) -> Optional[str]:
    """Extract main content from *url* as markdown.

    Strategy is determined by :func:`url_pattern_hint`:

    - ``arxiv_paper``  →  arXiv API (returns abstract as markdown)
    - ``hn_discussion``  →  HN Firebase API (post + top 30 comments)
    - ``reddit_discussion``  →  Reddit .json API (post + threaded comments)
    - ``discussion_site``  →  skip (return *None*)
    - ``tweet``  →  skip (return *None*; RSS summary is the fallback)
    - ``canonical_article``  →  trafilatura, then readability-lxml

    Returns the extracted markdown string, or *None* if all
    extraction methods fail or the URL is skipped.
    """
    if not url:
        logger.warning("fetch_fulltext called with empty url")
        return None

    hint = url_pattern_hint(url)
    logger.debug("Fetching fulltext from: %s (hint=%s)", url, hint)

    # Strategy: arXiv paper -> dedicated API
    if hint == "arxiv_paper":
        result = _fetch_arxiv_abstract(url)
        if result is not None:
            logger.debug("arXiv API succeeded for: %s", url)
            return result
        logger.warning("arXiv API failed for: %s", url)
        return None

    # Strategy: HN discussions -> Firebase API
    if hint == "hn_discussion":
        result = _fetch_hn_discussion(url)
        if result is not None:
            logger.debug("HN API succeeded for: %s", url)
            return result
        logger.warning("HN API failed for: %s", url)
        return None

    # Strategy: Reddit discussions -> .json API
    if hint == "reddit_discussion":
        result = _fetch_reddit_discussion(url)
        if result is not None:
            logger.debug("Reddit API succeeded for: %s", url)
            return result
        logger.warning("Reddit API failed for: %s", url)
        return None

    # Strategy: discussion sites (Mastodon etc.) / tweets -> skip extraction
    if hint in ("discussion_site", "tweet"):
        logger.debug("Skipping extraction for %s (hint=%s)", url, hint)
        return None

    # Strategy: canonical_article -> standard HTML extraction
    result = _extract_trafilatura(url, timeout=timeout)
    if result is not None:
        logger.debug("trafilatura succeeded for: %s", url)
        return result

    result = _extract_readability(url, timeout=timeout)
    if result is not None:
        logger.debug("readability-lxml fallback succeeded for: %s", url)
        return result

    logger.warning("All extraction methods failed for: %s", url)
    return None
