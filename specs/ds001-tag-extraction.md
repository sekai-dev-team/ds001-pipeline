# DS-001 Tag Extraction — Implementation Spec

> Add concept tag generation to DS-001's summarization step.
> Tags feed DS-004 v3.0's tag-based consolidation.

## Summary of Changes

DS-001 currently generates summaries. Now it also generates concept tags
during the same LLM call. Post-processing handles tag dedup and tag library
management. No additional LLM calls — the tagging is folded into the existing
Pass 2 summarization.

## Change 1: Expand SUMMARIZE_PROMPT (llm_filter.py L52-61)

Add concept tagging to the existing summarization prompt:

```python
SUMMARIZE_PROMPT = (
    "You are an AI news summarizer and classifier. For the article below, "
    "generate a Chinese summary with these structured sections:\n"
    "**主题:** [1 sentence — what the article is about]\n"
    "**方法/发现:** [2-3 sentences — key method, result, or insight]\n"
    "**意义:** [1 sentence — why it matters]\n"
    "Total: at least 150 Chinese characters, at most 500.\n\n"
    "CONCEPT TAGS: Assign 0-3 concept tags to this article. "
    "Known tags that are high-frequency:\n{known_tags}\n\n"
    "Rules:\n"
    "- Reuse an existing known tag name if the concept matches.\n"
    "- Create a NEW tag (kebab-case English name) if the topic is new.\n"
    "- Provide a one-sentence label_text for every tag.\n"
    "Format concept_tags as a JSON array: [{{\"name\": \"...\", \"label_text\": \"...\"}}]\n\n"
    'Return a JSON object with keys: ai_summary (string), concept_tags (array).\n'
    "Always return valid JSON."
)
```

The `{known_tags}` placeholder is filled with high-frequency tag names
(article_count > 5) + their label_text, e.g.:
```
- agent-memory: "AI agent memory and retrieval systems"
- rl-training: "reinforcement learning post-training methods"
```

## Change 2: Update _call_deepseek_summarize() (llm_filter.py)

- Add `known_tags: str` parameter
- Format the SUMMARIZE_PROMPT with the known_tags string
- Parse and return `concept_tags` from the LLM response alongside `ai_summary`
- Return type: `dict[str, Any] | None` with keys `ai_summary` and `concept_tags`
- `max_tokens` increased from 1024 → 1536 (to accommodate tags)

```python
def _call_deepseek_summarize(
    api_key: str, article_text: str, known_tags: str = ""
) -> dict[str, Any] | None:
    prompt = SUMMARIZE_PROMPT.format(known_tags=known_tags or "(no known tags yet)")
    # ... existing deepseek call logic ...
    # Parse response: extract ai_summary AND concept_tags
    # concept_tags defaults to [] if missing
```

## Change 3: Pass known_tags through filter_articles() (llm_filter.py)

```python
def filter_articles(
    articles: list[Article], 
    api_key: str,
    tag_library: dict[str, dict] | None = None,
) -> list[Article]:
```

In Pass 2 worker:
```python
# Build known_tags string
known = ""
if tag_library:
    high_freq = {k: v for k, v in tag_library.items() if v.get("article_count", 0) > 5}
    known = "\n".join(
        f'- {k}: "{v.get("label_text", "")}"' for k, v in sorted(high_freq.items())
    )

result = _call_deepseek_summarize(api_key, article_text, known_tags=known)
if result:
    article.ai_summary = result.get("ai_summary", "")
    article.concept_tags = result.get("concept_tags", [])
```

## Change 4: Tag Processing (knowledge_mcp.py)

New function `process_tags()` called after summarization, before writing notes:

```python
import json
import math
import requests  # already imported

def process_tags(articles: list[Article]) -> None:
    """For each article, process concept_tags: dedup against tag library,
    create new tags if needed, and set article.topic_tags for frontmatter.
    
    Modifies articles in-place: sets article.topic_tags (list of tag names).
    """
    tag_library = _read_tag_library()
    
    for article in articles:
        article.topic_tags = []
        if not hasattr(article, 'concept_tags') or not article.concept_tags:
            continue
        
        for tag in article.concept_tags:
            name = tag.get("name", "").strip().lower().replace(" ", "-")
            label = tag.get("label_text", "").strip()
            if not name or not label:
                continue
            
            if name in tag_library:
                # Existing tag: reuse
                article.topic_tags.append(name)
                tag_library[name]["article_count"] = tag_library[name].get("article_count", 0) + 1
                _update_tag_note(name, {
                    "article_count": tag_library[name]["article_count"],
                    "last_seen": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                })
            else:
                # New tag: embed and dedup
                embedding = _kmcp_embed(label)
                if not embedding:
                    # Fallback: just create the tag without embedding
                    _create_tag_note(name, label, [])
                    tag_library[name] = {
                        "name": name, "embedding": [], "label_text": label,
                        "article_count": 1, "concept_page": None,
                        "first_seen": date_str, "last_seen": date_str,
                    }
                    article.topic_tags.append(name)
                    continue
                
                # Cosine dedup against existing library tags
                best_match = None
                best_score = 0.0
                for lib_name, lib_info in tag_library.items():
                    lib_emb = lib_info.get("embedding")
                    if not lib_emb:
                        continue
                    score = _cosine_similarity(embedding, lib_emb)
                    if score > best_score:
                        best_score = score
                        best_match = lib_name
                
                if best_score > 0.90 and best_match:
                    # Semantic duplicate: reuse existing tag
                    article.topic_tags.append(best_match)
                    tag_library[best_match]["article_count"] = tag_library[best_match].get("article_count", 0) + 1
                    _update_tag_note(best_match, {
                        "article_count": tag_library[best_match]["article_count"],
                        "last_seen": date_str,
                    })
                else:
                    # Genuinely new tag
                    _create_tag_note(name, label, embedding)
                    tag_library[name] = {
                        "name": name, "embedding": embedding, "label_text": label,
                        "article_count": 1, "concept_page": None,
                        "first_seen": date_str, "last_seen": date_str,
                    }
                    article.topic_tags.append(name)
```

## Change 5: Frontmatter Injection (knowledge_mcp.py L270-277)

Add `topic/` tags to the frontmatter:

```python
# Build tags list with topic/ prefix
tags = ["ai-agent", "type/episodic", article.source_tag]
for tag_name in getattr(article, 'topic_tags', []):
    tags.append(f"topic/{tag_name}")

frontmatter = {
    "tags": tags,
    "memory_type": "episodic",
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "ingested_at": now_iso,
    "source_url": article.url,
    "has_fulltext": article.has_fulltext,
}
```

## Change 6: k-mcp Embed Client

Reuse the existing k-mcp client infrastructure (knowledge_mcp.py already has `_rpc_payload`, `_initialize`, `KMCP_BASE_URL`). Add:

```python
def _kmcp_embed(text: str) -> list[float] | None:
    """Call k-mcp embed tool and return embedding vector."""
    if not _initialize():
        return None
    payload = _rpc_payload("tools/call", {
        "name": "embed",
        "arguments": {"text": text},
    })
    try:
        resp = requests.post(KMCP_BASE_URL, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if "error" in result:
            return None
        content = result.get("result", {}).get("content", [])
        if content and isinstance(content, list):
            text_content = content[0].get("text", "[]")
            return json.loads(text_content)
    except Exception:
        pass
    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
```

## Change 7: Tag Library File I/O (knowledge_mcp.py)

```python
def _read_tag_library() -> dict[str, dict]:
    """Read all .md files in /vault/tags/, parse frontmatter."""
    tags_dir = os.path.join(VAULT_PATH, "tags")
    library = {}
    if not os.path.isdir(tags_dir):
        return library
    for fname in sorted(os.listdir(tags_dir)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(tags_dir, fname)
        try:
            with open(fpath, "r") as f:
                content = f.read()
        except OSError:
            continue
        fm = _parse_frontmatter(content)
        if not fm:
            continue
        name = fm.get("name") or fname[:-3]
        emb_raw = fm.get("embedding")
        if isinstance(emb_raw, str):
            try:
                emb_raw = json.loads(emb_raw)
            except (json.JSONDecodeError, TypeError):
                emb_raw = []
        library[name] = {
            "name": name, "embedding": emb_raw if isinstance(emb_raw, list) else [],
            "label_text": fm.get("label_text", ""),
            "article_count": fm.get("article_count", 0),
            "concept_page": fm.get("concept_page"),
            "first_seen": fm.get("first_seen"), "last_seen": fm.get("last_seen"),
        }
    return library


def _create_tag_note(name: str, label_text: str, embedding: list[float]) -> None:
    """Create /vault/tags/{name}.md."""
    tags_dir = os.path.join(VAULT_PATH, "tags")
    os.makedirs(tags_dir, exist_ok=True)
    emb_str = json.dumps(embedding)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = f"""---
name: {name}
embedding: {emb_str}
label_text: "{label_text}"
article_count: 1
concept_page:
first_seen: {date_str}
last_seen: {date_str}
---

# {name}

{label_text}
"""
    with open(os.path.join(tags_dir, f"{name}.md"), "w") as f:
        f.write(content)


def _update_tag_note(name: str, updates: dict) -> None:
    """Patch frontmatter of /vault/tags/{name}.md."""
    fpath = os.path.join(VAULT_PATH, "tags", f"{name}.md")
    # ... read, parse fm, apply updates, rewrite ...
```

## Change 8: Wire into collect.py

In `main()`, after Pass 2 summarization:
```python
# After Pass 2 filtering + summarization
articles = filter_articles(articles, api_key)  # existing call

# NEW: Tag processing
from pipeline.knowledge_mcp import process_tags
process_tags(articles)
```

No change needed in `write_notes_fs()` — it already reads article attributes
and will automatically include `topic/` tags from `article.topic_tags`.

## Implementation Order

1. Add `_kmcp_embed()`, `_cosine_similarity()` to knowledge_mcp.py
2. Add `_read_tag_library()`, `_create_tag_note()`, `_update_tag_note()` to knowledge_mcp.py
3. Add `process_tags()` to knowledge_mcp.py
4. Update SUMMARIZE_PROMPT in llm_filter.py
5. Update `_call_deepseek_summarize()` in llm_filter.py
6. Update `filter_articles()` signature and Pass 2 worker in llm_filter.py
7. Update frontmatter tags construction in `write_notes_fs()`
8. Wire `process_tags()` into collect.py
