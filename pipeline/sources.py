"""All 12 curated sources for DS-001 pipeline.

Every source function is wrapped in an independent try/except so a single
source failure never blocks the others.  Each returns a list of Article
objects or an empty list on error.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
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


def _within_last_24h(entry) -> bool:
    """Check if a feed entry was published within the last 24 hours.

    Uses ``published_parsed``, falling back to ``updated_parsed``.
    Returns *True* if no date info is available (better to include than
    silently drop articles with missing metadata).
    """
    parsed = (
        getattr(entry, "published_parsed", None)
        or getattr(entry, "updated_parsed", None)
    )
    if parsed is None:
        return True
    pub_time = datetime(*parsed[:6], tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - pub_time
    return age.total_seconds() <= 86400


def _rss_entry_to_article(entry, *, source_name: str, source_tag: str) -> Article | None:
    """Convert a feedparser entry to an Article, applying date filter.

    Returns *None* if the article is older than 24 hours.
    """
    if not _within_last_24h(entry):
        return None
    return Article(
        title=entry.get("title", ""),
        url=entry.get("link", ""),
        source_name=source_name,
        source_tag=source_tag,
        summary=_summary_from_entry(entry),
        published_at=_iso_date(entry.get("published_parsed")),
    )


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


# Track RSSHub Twitter warning so it only appears once
_rsshub_twitter_warned = False


def _warn_rsshub_twitter() -> None:
    global _rsshub_twitter_warned  # noqa: PLW0603
    if not _rsshub_twitter_warned:
        logger.warning("RSSHub Twitter routes unavailable, skipping 4 X sources")
        _rsshub_twitter_warned = True


# ====================== 1. Anthropic Research Blog ==========================

def _fetch_anthropic() -> list[Article]:
    logger.warning(
        "Anthropic Blog: no RSS feed available "
        "(all known feed URLs — /research/feed, /feed, /rss — return 404)"
    )
    return []

_register("Anthropic Blog", _fetch_anthropic)

# ====================== 2. Google DeepMind Blog ============================

def _fetch_deepmind() -> list[Article]:
    feed = _fetch_feed("https://deepmind.google/blog/feed")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        article = _rss_entry_to_article(
            entry, source_name="DeepMind Blog", source_tag="source/deepmind"
        )
        if article is not None:
            articles.append(article)
    return articles

_register("DeepMind Blog", _fetch_deepmind)

# ====================== 3. Hugging Face Blog ===============================

def _fetch_huggingface() -> list[Article]:
    feed = _fetch_feed("https://huggingface.co/blog/feed.xml")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        article = _rss_entry_to_article(
            entry, source_name="Hugging Face Blog", source_tag="source/huggingface"
        )
        if article is not None:
            articles.append(article)
    return articles

_register("Hugging Face Blog", _fetch_huggingface)

# ====================== 4. LangChain Engineering Blog ======================

def _fetch_langchain() -> list[Article]:
    logger.warning(
        "LangChain Blog: RSS feed unavailable "
        "(blog.langchain.dev/rss/ permanently redirects to www.langchain.com/blog HTML page; "
        "no RSS feed found on the new site)"
    )
    return []

_register("LangChain Blog", _fetch_langchain)

# ====================== 5. Qwen GitHub Releases ============================

def _fetch_qwen_releases() -> list[Article]:
    feed = _fetch_feed("https://github.com/QwenLM/Qwen/releases.atom")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        article = _rss_entry_to_article(
            entry, source_name="Qwen Releases", source_tag="source/qwen"
        )
        if article is not None:
            articles.append(article)
    return articles

_register("Qwen Releases", _fetch_qwen_releases)

# ====================== 6. DeepSeek GitHub Releases ========================

def _fetch_deepseek_releases() -> list[Article]:
    feed = _fetch_feed("https://github.com/deepseek-ai/DeepSeek-V3/releases.atom")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        article = _rss_entry_to_article(
            entry, source_name="DeepSeek Releases", source_tag="source/deepseek"
        )
        if article is not None:
            articles.append(article)
    return articles

_register("DeepSeek Releases", _fetch_deepseek_releases)

# ====================== 7. @_akhaliq (RSSHub X/Twitter) ===================

def _fetch_akhaliq() -> list[Article]:
    _warn_rsshub_twitter()
    return []

_register("@_akhaliq", _fetch_akhaliq)

# ====================== 8. @AnthropicAI (RSSHub X/Twitter) ================

def _fetch_anthropic_twitter() -> list[Article]:
    _warn_rsshub_twitter()
    return []

_register("@AnthropicAI", _fetch_anthropic_twitter)

# ====================== 9. @hwchase17 (RSSHub X/Twitter) ==================

def _fetch_hwchase17() -> list[Article]:
    _warn_rsshub_twitter()
    return []

_register("@hwchase17", _fetch_hwchase17)

# ====================== 10. @steipete (RSSHub X/Twitter) ==================

def _fetch_steipete() -> list[Article]:
    _warn_rsshub_twitter()
    return []

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
            if not _within_last_24h(entry):
                continue
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
        if not _within_last_24h(entry):
            continue
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
