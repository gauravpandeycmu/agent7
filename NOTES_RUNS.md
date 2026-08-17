# Reproducible run notes

The canonical, reviewed traces are under `evidence/`. They contain ten base-run files because Queries C and F each require two processes, plus one custom comparison report.

## State-sensitive pairs

Run C1 and then C2 without clearing state. C1 stores the birthday and creates two reminder files; C2 starts a new agent process and answers from persisted Memory with zero tool calls.

Run F1 and then F2 without clearing state. F1 embeds 15 chunks from the five supplied papers. F2 starts a new process, reloads `state/index.faiss` plus `state/index_ids.json`, retrieves both the CoT and ReAct sources, and answers without re-indexing.

```bash
uv run python run_base_queries.py c1 --trace evidence/base_C1_latest.txt
uv run python run_base_queries.py c2 --trace evidence/base_C2_latest.txt

uv run python run_base_queries.py f1 --trace evidence/base_F1_latest.txt
uv run python run_base_queries.py f2 --trace evidence/base_F2_latest.txt
```

## Runtime state

The following files are generated locally and intentionally ignored by Git:

```text
state/memory.json
state/index.faiss
state/index_ids.json
state/artifacts/
sandbox/
usage.json
```

## Live-data notes

Queries A, B, and D use live network data. Search rankings, page content, and weather forecasts can change. Query B resolves “Saturday” relative to the day on which it is run. The committed evidence is the verified 17 August 2026 run and reports Saturday, 22 August 2026.

The canonical evidence directory has no incomplete-agent, traceback, or tool-execution failure markers. Ignored legacy files under `traces/` are from earlier development and are not part of the submission.
