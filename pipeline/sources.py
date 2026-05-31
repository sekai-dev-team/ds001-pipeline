"""All 9 curated sources for DS-001 pipeline.

Every source function is wrapped in an independent try/except so a single
source failure never blocks the others.  Each returns a list of Article
objects or an empty list on error.
"""

from __future__ import annotations

import logging
import re
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

REQUEST_TIMEOUT = 60  # seconds (arXiv slow XML responses)
USER_AGENT = (
    "DS-001-Pipeline/0.1 (+https://github.com/sekai-dev-team/ds001-pipeline)"
)

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _fetch_feed(url: str, timeout: int = REQUEST_TIMEOUT) -> feedparser.FeedParserDict | None:
    """Fetch and parse an RSS/Atom feed with feedparser.

    Returns the parsed feed dict, or *None* on any failure.
    Retries up to 3 times on HTTP 429 (rate limit) using Retry-After header,
    and on transient errors with exponential backoff.
    """
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            resp = _session.get(url, timeout=timeout)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "15"))
                if attempt < max_retries:
                    logger.warning(
                        "HTTP 429 from %s, retrying in %ds (attempt %d/%d)",
                        url, retry_after, attempt + 1, max_retries,
                    )
                    time.sleep(retry_after)
                    continue
                logger.error("HTTP 429 from %s after %d retries", url, max_retries)
                return None
            resp.raise_for_status()
            return feedparser.parse(resp.content)
        except Exception:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
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


def _within_last_24h(entry, max_age_hours: int = 24) -> bool:
    """Check if a feed entry was published within *max_age_hours*.

    Uses ``published_parsed``, falling back to ``updated_parsed``.
    Returns *True* if no date info is available (better to include than
    silently drop articles with missing metadata).

    Pass ``max_age_hours=48`` for slow-publishing sources (HF Blog, etc.).
    """
    parsed = (
        getattr(entry, "published_parsed", None)
        or getattr(entry, "updated_parsed", None)
    )
    if parsed is None:
        return True
    pub_time = datetime(*parsed[:6], tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - pub_time
    return age.total_seconds() <= max_age_hours * 3600


def _nitter_to_x_url(url: str) -> str:
    """Convert a nitter.net URL to the equivalent x.com URL.

    Pattern: ``https://nitter.net/username/status/123456#m``
    → ``https://x.com/username/status/123456``

    Also handles plain profile links: ``https://nitter.net/username``
    → ``https://x.com/username``
    """
    # Specific: status links with optional fragment (#m, etc.)
    url = re.sub(
        r"^https?://nitter\.net/([^/]+)/status/(\d+)(?:#.*)?$",
        r"https://x.com/\1/status/\2",
        url,
    )
    # Catch-all: any other nitter.net path → same path on x.com
    url = re.sub(r"^https?://nitter\.net/", "https://x.com/", url)
    return url


def _rss_entry_to_article(entry, *, source_name: str, source_tag: str, max_age_hours: int = 24) -> Article | None:
    """Convert a feedparser entry to an Article, applying date filter.

    Returns *None* if the article is older than *max_age_hours*.
    """
    if not _within_last_24h(entry, max_age_hours=max_age_hours):
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


# Nitter instances for X/Twitter RSS (primary + fallbacks)
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.1d4.us",
    "https://nitter.space",
    "https://nitter.lacontrevoie.fr",
    "https://nitter.cz",
]


def _fetch_nitter_feed(handle: str, source_name: str, source_tag: str) -> list[Article]:
    """Fetch tweets for a given X/Twitter handle via Nitter RSS.

    Tries each Nitter instance in order. Returns articles on the first
    successful response, or an empty list if all instances fail.
    """
    for instance in NITTER_INSTANCES:
        url = f"{instance}/{handle}/rss"
        logger.debug("Trying Nitter feed: %s", url)
        feed = _fetch_feed(url)
        if feed is not None:
            articles: list[Article] = []
            for entry in feed.entries:
                article = _rss_entry_to_article(
                    entry, source_name=source_name, source_tag=source_tag
                )
                if article is not None:
                    # Convert nitter.net → x.com so source_url in frontmatter
                    # points to a live Twitter/X link instead of a dead proxy.
                    article.url = _nitter_to_x_url(article.url)
                    articles.append(article)
            logger.info(
                "Nitter feed '%s' (%s): %d articles",
                handle, instance, len(articles),
            )
            return articles
        logger.warning("Nitter instance %s failed for %s, trying next", instance, handle)

    logger.warning("All Nitter instances failed for %s — source unavailable", handle)
    return []


# ====================== 2. Google AI Blog ============================

def _fetch_google_ai() -> list[Article]:
    feed = _fetch_feed("https://blog.google/innovation-and-ai/technology/ai/rss/")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        article = _rss_entry_to_article(
            entry, source_name="Google AI Blog", source_tag="source/google-ai"
        )
        if article is not None:
            articles.append(article)
    return articles

_register("Google AI Blog", _fetch_google_ai)

# ====================== 3. Hugging Face Blog ===============================

def _fetch_huggingface() -> list[Article]:
    feed = _fetch_feed("https://huggingface.co/blog/feed.xml")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        article = _rss_entry_to_article(
            entry, source_name="Hugging Face Blog", source_tag="source/huggingface",
            max_age_hours=48,
        )
        if article is not None:
            articles.append(article)
    return articles

_register("Hugging Face Blog", _fetch_huggingface)

# ====================== 4. LangChain GitHub Releases ========================

def _fetch_langchain() -> list[Article]:
    feed = _fetch_feed("https://github.com/langchain-ai/langchain/releases.atom")
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        article = _rss_entry_to_article(
            entry,
            source_name="LangChain Releases",
            source_tag="source/langchain",
            max_age_hours=72,
        )
        if article is not None:
            articles.append(article)
    return articles

_register("LangChain Releases", _fetch_langchain)

# ====================== 5. @_akhaliq (Nitter RSS) ============================

def _fetch_akhaliq() -> list[Article]:
    return _fetch_nitter_feed(
        "_akhaliq",
        source_name="@_akhaliq",
        source_tag="source/akhaliq",
    )

_register("@_akhaliq", _fetch_akhaliq)

# ====================== 6. Anthropic News (Google News RSS) ===================

def _fetch_anthropic_news() -> list[Article]:
    """Google News RSS for Anthropic — catches news coverage, partnerships, launches."""
    url = "https://news.google.com/rss/search?q=Anthropic+Claude+AI&hl=en-US&gl=US&ceid=US:en"
    feed = _fetch_feed(url)
    if feed is None:
        return []
    articles: list[Article] = []
    for entry in feed.entries:
        article = _rss_entry_to_article(
            entry,
            source_name="Anthropic News (Google News)",
            source_tag="source/anthropic-news",
        )
        if article is not None:
            articles.append(article)
    return articles

_register("Anthropic News", _fetch_anthropic_news)

# ====================== 7. @hwchase17 (Nitter RSS) ===========================

def _fetch_hwchase17() -> list[Article]:
    return _fetch_nitter_feed(
        "hwchase17",
        source_name="@hwchase17",
        source_tag="source/hwchase17",
    )

_register("@hwchase17", _fetch_hwchase17)

# ====================== 8. @steipete (Nitter RSS) ==========================

def _fetch_steipete() -> list[Article]:
    return _fetch_nitter_feed(
        "steipete",
        source_name="@steipete",
        source_tag="source/steipete",
    )

_register("@steipete", _fetch_steipete)

# ====================== 9. Hacker News (3 sub-streams) ====================

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

# ====================== 10. arXiv (cs.AI + cs.CL) =========================

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
        # arXiv does NOT use _within_last_24h():
        # - published timestamp = SUBMISSION time, not announcement
        # - papers are announced in daily batches (submitted by 14:00 ET, announced ~20:00 ET next day)
        # - a 24h window from submission time means papers >24h old are invisible
        # - instead, we rely on sortBy=submittedDate desc + max_results=50 to get the latest batch
        # - k-mcp write_note dedup prevents re-ingestion of same URL

        title = entry.get("title", "").strip().replace("\n", " ")
        summary = _summary_from_entry(entry)

        # Post-hoc source tagging: check title/summary for Qwen/DeepSeek
        text_lower = (title + " " + summary).lower()
        if "qwen" in text_lower:
            source_name = "Qwen Papers (arXiv)"
            source_tag = "source/qwen"
        elif "deepseek" in text_lower:
            source_name = "DeepSeek Papers (arXiv)"
            source_tag = "source/deepseek"
        else:
            source_name = "arXiv"
            source_tag = "source/arxiv"

        articles.append(
            Article(
                title=title,
                url=link,
                source_name=source_name,
                source_tag=source_tag,
                summary=summary,
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
