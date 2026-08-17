# Session 7 run notes

Document state on disk after the official query sequence (for Query F cross-run demo).

## Expected layout after F run 1 → F run 2

```
state/
  memory.json      # facts + chunk facts + outcomes
  index.faiss      # 768-d vectors
  index_ids.json   # mem: ids parallel to FAISS rows
  artifacts/       # web fetch artifacts from A–D; RAG search artifacts from E–H
```

## Query order for reproducible F persistence

1. `--reset F1` (indexes all `papers/*.md`, reports chunk count — **5** after chrome-strip)
2. `F2` **without** `--reset` (fresh process OK; `state/` must remain)

Gateway: `llm_gatewayV7` on `:8107` (professor ZIP). Embeddings: `nomic-embed-text` 768-d.

The 51-chunk deliverable is `uv run python build_index.py --reset` (papers + Formula One `corpus/`). See `CORPUS_MANIFEST.md` and `README.md` for traces.
