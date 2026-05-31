"""Tests for _build_rss_fetcher — produces fetcher closures for RSS sources."""
from unittest.mock import patch, MagicMock
from pipeline.sources import _build_rss_fetcher
from pipeline.article import Article


def test_rss_builder_returns_callable():
    """_build_rss_fetcher() must return a callable."""
    fn = _build_rss_fetcher("Test Source", "source/test", "https://example.com/rss", 24)
    assert callable(fn)


def test_rss_builder_passes_correct_window():
    """The fetcher must pass max_age_hours to _rss_entry_to_article."""
    fn = _build_rss_fetcher("Test Source", "source/test", "https://example.com/rss", 72)

    mock_entry = MagicMock()
    mock_entry.title = "Test Article"
    mock_entry.link = "https://example.com/article"

    mock_feed = MagicMock()
    mock_feed.entries = [mock_entry]

    mock_article = MagicMock(spec=Article)

    with patch("pipeline.sources._fetch_feed", return_value=mock_feed):
        with patch("pipeline.sources._rss_entry_to_article", return_value=mock_article) as mock_convert:
            result = fn()

    assert result == [mock_article]
    mock_convert.assert_called_once_with(
        mock_entry,
        source_name="Test Source",
        source_tag="source/test",
        max_age_hours=72,
    )


def test_rss_builder_returns_empty_on_fetch_failure():
    """The fetcher must return [] when _fetch_feed returns None."""
    fn = _build_rss_fetcher("Test Source", "source/test", "https://example.com/rss", 24)
    with patch("pipeline.sources._fetch_feed", return_value=None):
        result = fn()
    assert result == []
