"""All 12 curated sources for DS-001 pipeline.

Every source function is wrapped in an independent try/except so a single
source failure never blocks the others.  Each returns a list of Article
objects or an empty list on error.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable

import feedparser
import requests

from pipeline.article import Article

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = (
    "DS-001-Pipeline/0.1 (+https://github.com/sekai-dev-team/ds001-pipeline)"
)

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _fetch_feed(url: str, timeout: int = REQUEST_TIMEOUT) -> feedparser.FeedParserDict | None:
    """Fetch and parse an RSS/Atom feed with feedparser.

    Returns the parsed feed dict, or *None* on any failure.
    """
    try:
        resp = _session.get(url, timeout=timeout)
        resp.raise_for_status()
        return feedparser.parse(resp.content)
    except Exception:
        logger.exception("Failed to fetch feed: %s", url)
        return None


def _iso_date(parsed_struct: time.struct_time | None) -> str:
    """Convert a feedparser time struct to ISO-8601 string."""
    if parsed_struct is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime(*parsed_struct[:6], tzinfo=timezone.utc).isoformat()


def _text_or_fallback(tag) -> str:
    """Extract clean text from a feedparser tag, or empty string."""
    if tag is None:
        return ""
    if hasattr(tag, "value"):
        return tag.value or ""
    if isinstance(tag, str):
        return tag
    return str(tag) if tag else ""


def _summary_from_entry(entry) -> str:
    """Extract summary text from a feedparser entry."""
    for attr in ("summary", "description", "content", "subtitle"):
        val = getattr(entry, attr, None)
        if val:
            if isinstance(val, list):
                return _text_or_fallback(val[0])[:300]
            return _text_or_fallback(val)[:300]
    return ""


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SourceFunc = Callable[[], list[Article]]
_source_registry: list[tuple[str, SourceFunc]] = []


def _register(name: str, fn: SourceFunc) -> None:
    _source_registry.append((name, fn))


def all_sources() -> list[tuple[str, SourceFunc]]:
    """Return list of (source_name, fetcher_fn) for every registered source."""
    return list(_source_registry)


# ====================== 1. Anthropic Research Blog ==========================

def _fetch_anthropic() -> list[Article]:
    feed = _fetch_feed("https://www.anthropic.com/research/feed")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="Anthropic Blog",
                source_tag="source/anthropic",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("Anthropic Blog", _fetch_anthropic)

# ====================== 2. Google DeepMind Blog ============================

def _fetch_deepmind() -> list[Article]:
    feed = _fetch_feed("https://deepmind.google/blog/feed")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="DeepMind Blog",
                source_tag="source/deepmind",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("DeepMind Blog", _fetch_deepmind)

# ====================== 3. Hugging Face Blog ===============================

def _fetch_huggingface() -> list[Article]:
    feed = _fetch_feed("https://huggingface.co/blog/feed.xml")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="Hugging Face Blog",
                source_tag="source/huggingface",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("Hugging Face Blog", _fetch_huggingface)

# ====================== 4. LangChain Engineering Blog ======================

def _fetch_langchain() -> list[Article]:
    feed = _fetch_feed("https://blog.langchain.dev/rss/")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="LangChain Blog",
                source_tag="source/langchain",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("LangChain Blog", _fetch_langchain)

# ====================== 5. Qwen GitHub Releases ============================

def _fetch_qwen_releases() -> list[Article]:
    feed = _fetch_feed("https://github.com/QwenLM/Qwen/releases.atom")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="Qwen Releases",
                source_tag="source/qwen",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("Qwen Releases", _fetch_qwen_releases)

# ====================== 6. DeepSeek GitHub Releases ========================

def _fetch_deepseek_releases() -> list[Article]:
    feed = _fetch_feed("https://github.com/deepseek-ai/DeepSeek-V3/releases.atom")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="DeepSeek Releases",
                source_tag="source/deepseek",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("DeepSeek Releases", _fetch_deepseek_releases)

# ====================== 7. @_akhaliq (RSSHub X/Twitter) ===================

def _fetch_akhaliq() -> list[Article]:
    feed = _fetch_feed("https://rsshub.app/twitter/user/_akhaliq")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="@_akhaliq",
                source_tag="source/akhaliq",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("@_akhaliq", _fetch_akhaliq)

# ====================== 8. @AnthropicAI (RSSHub X/Twitter) ================

def _fetch_anthropic_twitter() -> list[Article]:
    feed = _fetch_feed("https://rsshub.app/twitter/user/AnthropicAI")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="@AnthropicAI",
                source_tag="source/anthropic-twitter",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("@AnthropicAI", _fetch_anthropic_twitter)

# ====================== 9. @hwchase17 (RSSHub X/Twitter) ==================

def _fetch_hwchase17() -> list[Article]:
    feed = _fetch_feed("https://rsshub.app/twitter/user/hwchase17")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="@hwchase17",
                source_tag="source/hwchase17",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("@hwchase17", _fetch_hwchase17)

# ====================== 10. @steipete (RSSHub X/Twitter) ==================

def _fetch_steipete() -> list[Article]:
    feed = _fetch_feed("https://rsshub.app/twitter/user/steipete")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        articles.append(
            Article(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source_name="@steipete",
                source_tag="source/steipete",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("@steipete", _fetch_steipete)

# ====================== 11. Hacker News (3 sub-streams) ====================

HN_STREAMS = [
    ("HN Frontpage", "https://hnrss.org/frontpage?count=30"),
    ("HN Newest (points≥50)", "https://hnrss.org/newest?points=50"),
    ("HN Search (LLM/Agent)", "https://hnrss.org/newest?q=LLM+OR+Agent+OR+Claude+OR+open+source"),
]


def _fetch_hackernews() -> list[Article]:
    all_articles: list[Article] = []
    seen_urls: set[str] = set()
    for stream_name, url in HN_STREAMS:
        feed = _fetch_feed(url)
        if feed is None:
            logger.warning("HN stream '%s' failed, continuing", stream_name)
            continue
        for entry in feed.entries:
            link = entry.get("link", "")
            if link in seen_urls:
                continue
            seen_urls.add(link)
            all_articles.append(
                Article(
                    title=entry.get("title", ""),
                    url=link,
                    source_name=f"Hacker News ({stream_name})",
                    source_tag="source/hackernews",
                    summary=_summary_from_entry(entry),
                    published_at=_iso_date(entry.get("published_parsed")),
                )
            )
    return all_articles

_register("Hacker News", _fetch_hackernews)

# ====================== 12. arXiv (cs.AI + cs.CL) =========================

ARXIV_QUERY = (
    "http://export.arxiv.org/api/query?"
    "search_query=cat:cs.AI+OR+cat:cs.CL"
    "&sortBy=submittedDate&sortOrder=descending&max_results=50"
)


def _fetch_arxiv() -> list[Article]:
    feed = _fetch_feed(ARXIV_QUERY)
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        # arXiv entries have id like "http://arxiv.org/abs/1234.56789v1"
        link = entry.get("id", "") or entry.get("link", "")
        # arXiv links in <id> are http, prefer https
        if link.startswith("http://"):
            link = "https://" + link[7:]
        articles.append(
            Article(
                title=entry.get("title", "").strip().replace("\n", " "),
                url=link,
                source_name="arXiv",
                source_tag="source/arxiv",
                summary=_summary_from_entry(entry),
                published_at=_iso_date(entry.get("published_parsed")),
            )
        )
    return articles

_register("arXiv", _fetch_arxiv)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_source(name: str) -> list[Article]:
    """Fetch a single named source.

    Returns a list of Articles (possibly empty on error).
    """
    for sname, sfunc in _source_registry:
        if sname == name:
            try:
                return sfunc()
            except Exception:
                logger.exception("Unexpected error fetching source '%s'", name)
                return []
    logger.warning("Unknown source: %s", name)
    return []


def fetch_all(mode: str = "daily") -> dict[str, tuple[list[Article], str | None]]:
    """Fetch all sources for the given *mode*.

    Returns ``{source_name: (articles, error_message)}``.
    Every source is wrapped in try/except — one failure never blocks others.
    """
    results: dict[str, tuple[list[Article], str | None]] = {}

    if mode == "hn-arxiv":
        target = ["Hacker News", "arXiv"]
    else:
        target = [sname for sname, _ in _source_registry]

    for sname, sfunc in _source_registry:
        if sname not in target:
            continue
        try:
            articles = sfunc()
            results[sname] = (articles, None)
            logger.info("Source '%s': %d articles", sname, len(articles))
        except Exception as exc:
            logger.exception("Source '%s' failed", sname)
            results[sname] = ([], str(exc))

    return results
