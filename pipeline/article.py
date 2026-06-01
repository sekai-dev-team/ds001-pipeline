from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Article:
    """Normalized article representation for all sources."""

    title: str
    url: str  # dedup key
    source_name: str  # human-readable, e.g. "Anthropic Blog"
    source_tag: str  # e.g. "source/anthropic"
    summary: str  # RSS description or first 300 chars
    published_at: str  # ISO 8601
    relevant: bool = False  # set by LLM filter
    ai_summary: str = ""  # set by LLM filter (2-3 sentences, Chinese)
    concept_tags: list[dict] = field(default_factory=list)  # set by Pass 2 summarization: [{"name": "...", "label_text": "..."}]
    topic_tags: list[str] = field(default_factory=list)  # set by process_tags(): deduped tag names for frontmatter
    fulltext: Optional[str] = None  # set by fulltext extraction (Step 3.5)
    has_fulltext: bool = False  # whether fulltext extraction succeeded
    content_type: Optional[str] = None  # set by LLM filter: article | discussion | link_only | low_quality

    def to_dict(self) -> dict:
        return asdict(self)
