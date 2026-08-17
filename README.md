# Assignment 7 — Memory, Retrieval, FAISS, and Indexed Knowledge

This repository extends the supplied Session 6 agent with the Session 7 retrieval path while keeping the same four-role loop:

```text
Memory.read -> Perception.observe -> Decision.next_step
            -> Action.execute -> Memory.record_outcome
```

Memory now embeds durable items, persists normalized vectors in FAISS, and falls back to keyword overlap only when vector retrieval produces no result. The MCP server adds document indexing and indexed search. Perception still describes goals without naming MCP tools.

## Submission status

The canonical evidence is in `evidence/`. Every saved run contains the verbatim query, stays within its assigned iteration bound, reaches `[done]`, and has a non-empty `FINAL` answer.

| Base query | Demonstration | Saved run | Iterations / bound |
|---|---|---:|---:|
| A | Fetch and extract Claude Shannon facts | [`base_A.txt`](evidence/base_A.txt) | 3 / 3 |
| B | Tokyo activities plus live Saturday weather | [`base_B.txt`](evidence/base_B.txt) | 5 / 8 |
| C | Persist birthday and create two reminders; retrieve in a new run | [`base_C1.txt`](evidence/base_C1.txt), [`base_C2.txt`](evidence/base_C2.txt) | 4 / 4, 2 / 3 |
| D | Search, open the top three results, and synthesize common advice | [`base_D.txt`](evidence/base_D.txt) | 6 / 6 |
| E | Index `attention.md` and return exactly three contributions | [`base_E.txt`](evidence/base_E.txt) | 4 / 5 |
| F | Index all five supplied papers, then query the persisted index | [`base_F1.txt`](evidence/base_F1.txt), [`base_F2.txt`](evidence/base_F2.txt) | 8 / 11, 3 / 3 |
| G | Semantic cross-paper retrieval for “credit assignment” | [`base_G.txt`](evidence/base_G.txt) | 3 / 4 |
| H | Compare ReAct with Chain-of-Thought | [`base_H.txt`](evidence/base_H.txt) | 3 / 3 |

The five supplied paper files produce 15 document chunks in Query F1 with the required 400-word size and 80-word overlap. The Formula One application is a separate 131-item corpus.

Run the deterministic submission checker before recording or submitting:

```bash
uv run python verify_assignment.py
```

It checks the source files, the Perception gate, all ten base-run trace files, iteration bounds, final-answer completion, the 131-item corpus, both semantic lexical-absence cases, and the five-with/five-without custom comparison.

## Setup

Requirements:

- Python 3.11–3.13 and `uv`
- Ollama with `qwen2.5:7b-instruct` and `nomic-embed-text`
- Internet access for the live web and weather demonstrations (A, B, and D)

```bash
cp .env.example .env
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
uv sync
```

No cloud-model key is required by the default configuration. Chat and embeddings both go through the supplied Gateway V7 on `http://localhost:8107`; `agent7.py` starts it when necessary. `TAVILY_API_KEY` is optional.

## Run the base demonstrations

Run all eight demonstrations in the required state order:

```bash
uv run python run_base_queries.py all --trace evidence/base_all_latest.txt
```

For a shorter recording, the persistence pair is especially useful:

```bash
uv run python run_base_queries.py f1 --trace evidence/base_F1_latest.txt
uv run python run_base_queries.py f2 --trace evidence/base_F2_latest.txt
```

F2 deliberately does not clear `state/`; it retrieves the FAISS index written by F1 in the preceding process. C1/C2 demonstrates the same persistence contract with a remembered birthday.

Queries A, B, and D use live sources, so URLs, forecasts, and wording can change when rerun. The committed evidence records the verified run from 17 August 2026; Query B correctly asks Open-Meteo for the next Saturday at runtime.

## Custom RAG application

The custom application indexes 131 explicit Formula One knowledge items from 13 Markdown files. Each overview or meaningful level-two section is one independently embedded item; the 75 season records are individual items. Boilerplate “Supplement” sections are excluded. See [`CORPUS_MANIFEST.md`](CORPUS_MANIFEST.md).

```bash
uv run python build_index.py --count-only
uv run python run_custom_queries.py --compare \
  --output evidence/custom_comparison.md
```

| ID | What it tests | Required indexed answer | Semantic |
|---|---|---|---:|
| FQ1 | 2003 neck-protection mandate | HANS / Head and Neck Support | No |
| FQ2 | Return of underfloor ground-effect tunnels | 2022 | No |
| FQ3 | Animal-like term for new-car vertical oscillation | porpoising | Yes |
| FQ4 | British driver who matched Schumacher’s record | Lewis Hamilton; seven each | Yes |
| FQ5 | Pit strategy terminology | undercut versus overcut | No |

The comparison report is [`evidence/custom_comparison.md`](evidence/custom_comparison.md): all five queries fail with an empty index and all five pass after indexing. FQ3 avoids the answer word `porpoising`; its probe phrase `rapid up-and-down motion` is absent from the corpus. FQ4 avoids both drivers’ names; its phrase `British competitor` is absent from the corpus.

## Architectural checks

- The Session 6 Memory → Perception → Decision → Action control flow remains intact.
- `MemoryItem.embedding` is optional, and durable facts/outcomes are embedded through Gateway V7.
- `state/index.faiss` and `state/index_ids.json` persist FAISS vectors and their memory-ID mapping.
- `index_document` and `search_knowledge` are exposed through MCP; Decision learns them from MCP metadata.
- `Perception.SYSTEM` contains zero MCP tool names. `verify_assignment.py` checks all twelve names currently exposed by the server.
- Artifact payloads stay outside Memory; Memory stores handles and bounded previews.
- Formula One corpus text and embeddings are pinned to local Ollama rather than sent to an external fallback.

The crawler uses Crawl4AI as its primary page reader. If local Chromium cannot start, the tool reports that condition and uses a requests/BeautifulSoup fallback so the live demonstration remains usable.

## Video outline

A concise 5–7 minute demo can follow this order:

1. Run `verify_assignment.py` and show that every contract check passes.
2. Briefly show the unchanged four-role loop and the Perception zero-tool-name gate.
3. Run F1 followed by F2 to demonstrate persisted document retrieval across processes.
4. Run the custom `--compare` command; highlight FQ3 and FQ4 as semantic queries.
5. Open the evidence table above and the corpus manifest, then show the GitHub repository.

**YouTube video:** [Assignment 7 demonstration](https://youtu.be/abD2VOAhqDo)

## Submission checklist

- [x] Eight verbatim base queries pass within their bounds
- [x] Corpus contains at least 50 real items (131 indexed items)
- [x] Five custom queries pass with the index and fail without it
- [x] At least two custom queries require semantic recall
- [x] Perception SYSTEM contains zero MCP tool names
- [x] README contains the corpus manifest links and saved traces
- [x] Add the YouTube URL
- [x] Push the final repository and confirm that `.env` and runtime `state/` are not committed

Only `evidence/` should be used as submission evidence. Any ignored files under the legacy `traces/` directory came from earlier development attempts and are intentionally excluded.
