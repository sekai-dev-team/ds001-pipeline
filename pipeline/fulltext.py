"""Full-text extraction from article URLs.

Primary: trafilatura (extract markdown from HTML).
Fallback: readability-lxml + html2text.
arXiv: dedicated API for paper abstracts.
Returns None if all methods fail (or URL is a known discussion site).
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

    Returns one of: ``"canonical_article"``, ``"discussion_site"``,
    ``"arxiv_paper"``, ``"tweet"``.
    """
    if not url:
        return "canonical_article"

    # arXiv abstract pages -> dedicated API
    if re.search(r"arxiv\.org/abs/\d+", url):
        return "arxiv_paper"

    # Nitter tweets (JS-rendered pages)
    if re.search(r"nitter\.net/\w+/status/", url):
        return "tweet"

    # Discussion sites — no original write-up
    if "news.ycombinator.com/item" in url or "old.reddit.com" in url:
        return "discussion_site"
    if ".mastodon." in url or "mastodon.social" in url:
        return "discussion_site"

    return "canonical_article"


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

    # Strategy: discussion sites / tweets -> skip extraction
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
