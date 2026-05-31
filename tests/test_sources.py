"""Tests for pipeline/sources.py — v0.2.10: Merge arXiv sub-queries + rate-limit fix."""

from unittest.mock import patch, MagicMock
import pytest
import pipeline.sources
from pipeline.sources import (
    _fetch_arxiv,
    _fetch_feed,
    all_sources,
    REQUEST_TIMEOUT,
)


# ====================== YAML-driven source registry ============================


def test_langchain_source_in_registry():
    """LangChain Releases must be registered via YAML with window=72."""
    sources = dict(all_sources())
    assert "LangChain Releases" in sources, \
        "LangChain Releases must be in the source registry"


# ====================== v0.2.10: REQUEST_TIMEOUT ==================================


def test_request_timeout_is_60s():
    """REQUEST_TIMEOUT must be 60s for arXiv's slow XML responses."""
    assert REQUEST_TIMEOUT == 60, "Should be 60s for arXiv slow responses"


# ====================== v0.2.10: 429 retry in _fetch_feed ========================


def test_fetch_feed_retries_on_429_then_succeeds():
    """_fetch_feed() must retry on HTTP 429 and succeed on retry."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "1"}
    mock_resp.raise_for_status.side_effect = None

    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    mock_resp_ok.raise_for_status.return_value = None

    mock_parsed = MagicMock()
    mock_parsed.entries = []

    with patch("pipeline.sources._session") as mock_session:
        mock_session.get.side_effect = [mock_resp, mock_resp_ok]
        with patch("pipeline.sources.feedparser.parse", return_value=mock_parsed):
            with patch("pipeline.sources.time.sleep") as mock_sleep:
                result = _fetch_feed("https://example.com/feed")

    assert result is mock_parsed, "Should return parsed feed after retry"
    assert mock_session.get.call_count == 2, "Should have made 2 requests (1 failed, 1 succeeded)"
    mock_sleep.assert_called_once_with(1)


def test_fetch_feed_returns_none_after_exhausting_429_retries():
    """_fetch_feed() must return None after exhausting 429 retries."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "1"}
    mock_resp.raise_for_status.side_effect = None

    with patch("pipeline.sources._session") as mock_session:
        mock_session.get.return_value = mock_resp
        with patch("pipeline.sources.time.sleep") as mock_sleep:
            result = _fetch_feed("https://example.com/feed")

    assert result is None, "Should return None after max 429 retries"
    assert mock_session.get.call_count == 4, "Should have tried 4 times (initial + 3 retries)"


def test_fetch_feed_uses_15s_default_retry_after_when_header_missing():
    """_fetch_feed() must default Retry-After to 15s when the header is absent."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {}  # No Retry-After header
    mock_resp.raise_for_status.side_effect = None

    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    mock_resp_ok.raise_for_status.return_value = None

    mock_parsed = MagicMock()
    mock_parsed.entries = []

    with patch("pipeline.sources._session") as mock_session:
        mock_session.get.side_effect = [mock_resp, mock_resp_ok]
        with patch("pipeline.sources.feedparser.parse", return_value=mock_parsed):
            with patch("pipeline.sources.time.sleep") as mock_sleep:
                result = _fetch_feed("https://example.com/feed")

    assert result is mock_parsed, "Should return parsed feed after retry"
    mock_sleep.assert_called_once_with(15)


def test_fetch_feed_retries_on_network_error_then_succeeds():
    """_fetch_feed() must retry on non-429 exceptions and succeed on retry."""
    with patch("pipeline.sources._session") as mock_session:
        mock_session.get.side_effect = [
            Exception("Connection error"),
            MagicMock(status_code=200),
        ]
        mock_parsed = MagicMock()
        mock_parsed.entries = []
        with patch("pipeline.sources.feedparser.parse", return_value=mock_parsed):
            with patch("pipeline.sources.time.sleep") as mock_sleep:
                result = _fetch_feed("https://example.com/feed")

    assert result is mock_parsed, "Should return parsed feed after retry on network error"
    assert mock_session.get.call_count == 2, "Should have made 2 requests"
    mock_sleep.assert_called_once()


def test_fetch_feed_returns_none_after_exhausting_error_retries():
    """_fetch_feed() must return None after all non-429 retries exhausted."""
    with patch("pipeline.sources._session") as mock_session:
        mock_session.get.side_effect = Exception("Persistent error")
        with patch("pipeline.sources.time.sleep"):
            result = _fetch_feed("https://example.com/feed")

    assert result is None, "Should return None after all retries exhausted"


# ====================== v0.2.10: Post-hoc tagging in _fetch_arxiv ================


def _make_arxiv_entry(title="Test Paper", summary="A test arxiv paper"):
    """Helper to create a mock arXiv entry."""
    entry = MagicMock()
    entry.get.side_effect = lambda key, default=None, _t=title, _s=summary: {
        "id": "http://arxiv.org/abs/1234.56789",
        "title": _t,
        "link": "",
    }.get(key, default)
    entry.title = title
    return entry


def test_fetch_arxiv_tags_qwen_papers():
    """_fetch_arxiv() must tag papers with 'qwen' in title as Qwen."""
    entry = MagicMock()
    entry.get.side_effect = lambda key, default=None: {
        "id": "http://arxiv.org/abs/1234.56789",
        "title": "Qwen2.5: A New Language Model",
        "link": "",
        "published_parsed": None,
    }.get(key, default)
    entry.title = "Qwen2.5: A New Language Model"

    mock_feed = MagicMock()
    mock_feed.entries = [entry]

    with patch("pipeline.sources._fetch_feed", return_value=mock_feed):
        with patch("pipeline.sources._summary_from_entry", return_value="A paper about the Qwen2.5 model"):
            articles = _fetch_arxiv()

    assert len(articles) == 1
    assert articles[0].source_name == "Qwen Papers (arXiv)"
    assert articles[0].source_tag == "source/qwen"


def test_fetch_arxiv_tags_deepseek_papers():
    """_fetch_arxiv() must tag papers with 'deepseek' in summary as DeepSeek."""
    entry = MagicMock()
    entry.get.side_effect = lambda key, default=None: {
        "id": "http://arxiv.org/abs/9876.54321",
        "title": "A Novel Architecture",
        "link": "",
        "published_parsed": None,
    }.get(key, default)
    entry.title = "A Novel Architecture"

    mock_feed = MagicMock()
    mock_feed.entries = [entry]

    with patch("pipeline.sources._fetch_feed", return_value=mock_feed):
        with patch("pipeline.sources._summary_from_entry", return_value="This paper introduces DeepSeek-v3"):
            articles = _fetch_arxiv()

    assert len(articles) == 1
    assert articles[0].source_name == "DeepSeek Papers (arXiv)"
    assert articles[0].source_tag == "source/deepseek"


def test_fetch_arxiv_tags_other_papers_as_arxiv():
    """_fetch_arxiv() must tag papers without Qwen/DeepSeek keywords as arXiv."""
    entry = MagicMock()
    entry.get.side_effect = lambda key, default=None: {
        "id": "http://arxiv.org/abs/1111.2222",
        "title": "Generic ML Paper",
        "link": "",
        "published_parsed": None,
    }.get(key, default)
    entry.title = "Generic ML Paper"

    mock_feed = MagicMock()
    mock_feed.entries = [entry]

    with patch("pipeline.sources._fetch_feed", return_value=mock_feed):
        with patch("pipeline.sources._summary_from_entry", return_value="Some machine learning research"):
            articles = _fetch_arxiv()

    assert len(articles) == 1
    assert articles[0].source_name == "arXiv"
    assert articles[0].source_tag == "source/arxiv"


# ====================== v0.2.10: Qwen/DeepSeek removed from registry =============


def test_qwen_not_in_source_registry():
    """Qwen Papers (arXiv) must be removed from the source registry."""
    source_names = [name for name, _ in pipeline.sources.all_sources()]
    assert "Qwen Papers (arXiv)" not in source_names


def test_deepseek_not_in_source_registry():
    """DeepSeek Papers (arXiv) must be removed from the source registry."""
    source_names = [name for name, _ in pipeline.sources.all_sources()]
    assert "DeepSeek Papers (arXiv)" not in source_names


def test_qwen_function_not_callable():
    """_fetch_qwen_releases() must be deleted."""
    assert not hasattr(pipeline.sources, "_fetch_qwen_releases"), \
        "_fetch_qwen_releases should not exist in sources module"


def test_deepseek_function_not_callable():
    """_fetch_deepseek_releases() must be deleted."""
    assert not hasattr(pipeline.sources, "_fetch_deepseek_releases"), \
        "_fetch_deepseek_releases should not exist in sources module"
