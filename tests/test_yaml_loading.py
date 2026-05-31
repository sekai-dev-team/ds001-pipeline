"""Tests for YAML source configuration loading."""
from pipeline.sources import _load_yaml_sources


def test_load_yaml_returns_list():
    """_load_yaml_sources() must return a list of source configs."""
    sources = _load_yaml_sources()
    assert isinstance(sources, list)
    assert len(sources) > 0


def test_load_yaml_required_fields():
    """Every source config must have name, tag, and window fields."""
    sources = _load_yaml_sources()
    for src in sources:
        assert "name" in src, f"Missing 'name' in {src}"
        assert "tag" in src, f"Missing 'tag' in {src}"
        assert "window" in src, f"Missing 'window' in {src}"
        assert isinstance(src["window"], int), f"'window' must be int in {src}"


def test_load_yaml_either_url_nitter_or_hn_streams():
    """Each source must have exactly one of: url, nitter, hn_streams."""
    sources = _load_yaml_sources()
    for src in sources:
        has_url = "url" in src
        has_nitter = "nitter" in src
        has_hn = "hn_streams" in src
        assert sum([has_url, has_nitter, has_hn]) == 1, \
            f"Source {src['name']} must have exactly one of url/nitter/hn_streams, got url={has_url} nitter={has_nitter} hn_streams={has_hn}"


def test_load_yaml_contains_expected_sources():
    """YAML must contain the expected source names."""
    sources = _load_yaml_sources()
    names = [s["name"] for s in sources]
    assert "Google AI Blog" in names
    assert "HuggingFace Blog" in names
    assert "LangChain Releases" in names
    assert "Anthropic News" in names
    assert "@_akhaliq" in names
    assert "@hwchase17" in names
    assert "@steipete" in names
    assert "Hacker News" in names
    assert len(names) == 8, f"Expected 8 sources, got {len(names)}"


def test_load_yaml_hn_streams_is_list():
    """Hacker News config must have hn_streams as a list of URLs."""
    sources = _load_yaml_sources()
    hn = [s for s in sources if s["name"] == "Hacker News"][0]
    assert "hn_streams" in hn
    assert isinstance(hn["hn_streams"], list)
    assert len(hn["hn_streams"]) == 3


def test_load_yaml_nitter_has_valid_handle():
    """Nitter configs must have a non-empty nitter handle."""
    sources = _load_yaml_sources()
    for src in sources:
        if "nitter" in src:
            assert isinstance(src["nitter"], str) and len(src["nitter"]) > 0
