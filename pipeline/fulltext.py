"""Full-text extraction from article URLs.

Primary: trafilatura (extract markdown from HTML).
Fallback: readability-lxml + html2text.
Returns None if all methods fail.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

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

    Uses trafilatura as the primary method, falling back to
    readability-lxml + html2text if trafilatura fails or returns
    no content.

    Returns the extracted markdown string, or *None* if all
    extraction methods fail.
    """
    if not url:
        logger.warning("fetch_fulltext called with empty url")
        return None

    logger.debug("Fetching fulltext from: %s", url)

    # Primary: trafilatura
    result = _extract_trafilatura(url, timeout=timeout)
    if result is not None:
        logger.debug("trafilatura succeeded for: %s", url)
        return result

    # Fallback: readability-lxml + html2text
    result = _extract_readability(url, timeout=timeout)
    if result is not None:
        logger.debug("readability-lxml fallback succeeded for: %s", url)
        return result

    logger.warning("All extraction methods failed for: %s", url)
    return None
