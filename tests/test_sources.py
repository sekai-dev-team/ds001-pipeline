"""Tests for pipeline/sources.py — LangChain Releases 72h fix."""

from unittest.mock import patch, MagicMock
from pipeline.sources import _fetch_langchain


def test_fetch_langchain_passes_max_age_72h():
    """_fetch_langchain() must pass max_age_hours=72 to _rss_entry_to_article
    so that releases published every 2-3 days are not filtered out."""
    # Mock _fetch_feed to return a feed with one dummy entry
    mock_entry = MagicMock()
    mock_entry.title = "Test Release v0.1.0"
    mock_entry.link = "https://github.com/langchain-ai/langchain/releases/tag/v0.1.0"
    mock_feed = MagicMock()
    mock_feed.entries = [mock_entry]

    mock_article = MagicMock()

    with patch("pipeline.sources._fetch_feed", return_value=mock_feed), \
         patch("pipeline.sources._rss_entry_to_article", return_value=mock_article) as mock_to_article:

        result = _fetch_langchain()

        # Should get back the mock article
        assert result == [mock_article], "Should return the article from _rss_entry_to_article"

        # _rss_entry_to_article must be called with max_age_hours=72
        mock_to_article.assert_called_once_with(
            mock_entry,
            source_name="LangChain Releases",
            source_tag="source/langchain",
            max_age_hours=72,
        )
