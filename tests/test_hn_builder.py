"""Tests for _build_hn_fetcher — multi-stream HN source."""
from unittest.mock import patch, MagicMock
from pipeline.sources import _build_hn_fetcher
from pipeline.article import Article


def test_hn_builder_returns_callable():
    """_build_hn_fetcher() must return a callable."""
    fn = _build_hn_fetcher("Hacker News", "source/hackernews", 24, [
        "https://hnrss.org/frontpage?count=30",
        "https://hnrss.org/newest?points=50",
    ])
    assert callable(fn)


def test_hn_builder_deduplicates_urls():
    """The fetcher must deduplicate articles with the same URL across streams."""
    fn = _build_hn_fetcher("Hacker News", "source/hackernews", 24, [
        "https://hnrss.org/frontpage?count=30",
        "https://hnrss.org/newest?points=50",
    ])

    mock_entry_1 = MagicMock()
    mock_entry_1.get.side_effect = lambda key, default=None: {
        "link": "https://news.ycombinator.com/item?id=123",
        "title": "Same Article",
    }.get(key, default)

    mock_entry_2 = MagicMock()
    mock_entry_2.get.side_effect = lambda key, default=None: {
        "link": "https://news.ycombinator.com/item?id=456",
        "title": "Different Article",
    }.get(key, default)

    # Both feeds return the same article + a different one
    feed_1 = MagicMock()
    feed_1.entries = [mock_entry_1, mock_entry_2]

    feed_2 = MagicMock()
    feed_2.entries = [mock_entry_1]  # Duplicate URL

    with patch("pipeline.sources._fetch_feed") as mock_fetch_feed:
        mock_fetch_feed.side_effect = [feed_1, feed_2]
        with patch("pipeline.sources._within_last_24h", return_value=True):
            with patch("pipeline.sources._summary_from_entry", return_value="summary"):
                with patch("pipeline.sources._iso_date", return_value="2025-01-01T00:00:00"):
                    articles = fn()

    # Should only have 2 unique articles (not 3)
    assert len(articles) == 2
    urls = [a.url for a in articles]
    assert "https://news.ycombinator.com/item?id=123" in urls
    assert "https://news.ycombinator.com/item?id=456" in urls
