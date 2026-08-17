# Formula One custom RAG corpus

These 13 Markdown documents become 131 explicit knowledge items. `build_index.py` indexes each document overview and meaningful level-two section separately; boilerplate Supplement sections are excluded.

```bash
uv run python build_index.py --count-only
# 131

uv run python run_custom_queries.py --compare \
  --output evidence/custom_comparison.md
```

See `CORPUS_MANIFEST.md` for the per-file item counts and semantic-query proof.
