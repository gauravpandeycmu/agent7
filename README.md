# agent7

Assignment 7 — the Session 6 **Perception / Memory / Decision / Action** loop, plus Session 7 **vector-first RAG**. Not a new agent. Not LangChain. Not a summariser glued onto FAISS.

Queries **A–D** are the Session 6 carry-over. Queries **E–H** are the RAG queries. Five extra Formula One queries (`FQ1`–`FQ5`) are the custom RAG app.

## What changed from Assignment 6

The four-role loop is the same file layout and the same contracts. Session 7 only extends Memory and MCP:

```
agent7 loop
  → Memory.read          FAISS first (768-d, IndexFlatIP + L2-norm).
                         Keyword overlap only if FAISS returns nothing.
  → Perception.observe   Gemini, structured JSON goals. SYSTEM prompt has
                         zero MCP tool names (grep gate).
  → Artifacts.get_bytes  if the open goal has attach=
  → Decision.next_step   Ollama to pick a tool; Gemini when bytes are attached
  → Action.execute       MCP stdio — Session 6 nine tools + index_document
                         + search_knowledge
  → Memory.record_outcome
  → append history, iterate
```

Every LLM and embedding call goes through **LLM Gateway V7** at `http://localhost:8107`. Chat is the V3-compatible `/v1/chat`. Embeddings are `POST /v1/embed` (Ollama `nomic-embed-text`, 768-d). There are no provider SDKs in this repo.

| Layer | Session 6 | Session 7 |
|---|---|---|
| Memory.read | keyword overlap | **vector first**; keyword fallback only if FAISS is empty |
| MemoryItem | facts / preferences / outcomes | optional `embedding: list[float] \| None` |
| MCP | 9 tools | + `index_document(path, chunk_size=400, overlap=80)`, `search_knowledge(query, k)` |
| Persist | `state/memory.json` | + `state/index.faiss`, `state/index_ids.json` |
| Gateway | V3 `:8101` | **V7 `:8107`** (professor ZIP in `llm_gatewayV7/`) |

`search_knowledge` returns a **dict** `{query, count, chunks}` (a raw list serializes empty through FastMCP). Perception never names either new tool. Decision picks them. Action is the only MCP client.

Chunking: 400 words, 80-word overlap. arXiv HTML chrome is stripped so the Abstract is what gets embedded. Query G’s phrase **“credit assignment” is absent** from every paper file — it is a synonym / vector recall, not a keyword hit.

## Setup

Needs **uv**, **Ollama** with `qwen2.5:7b-instruct` (Decision tool-picking) and `nomic-embed-text` (embeddings), and a **Gemini** key (Perception, Memory classification, Decision-with-attachments).

```bash
cp .env.example .env          # paste GEMINI_API_KEY
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
uv sync
uv run playwright install chromium
```

`TAVILY_API_KEY` is optional. Without it, `web_search` uses DuckDuckGo.

Gateway V7 starts on `:8107` automatically from `agent7.py` (`llm.ensure_gateway()`). Dashboard: http://localhost:8107

Wipe between attempts:

```bash
uv run python agent7.py --clean
```

## Run the base queries (A–H)

```bash
uv run python agent7.py --reset A     # Shannon Wikipedia (artifact attach)
uv run python agent7.py --reset B     # Tokyo + Saturday weather
uv run python agent7.py --reset C1    # remember birthday + reminder files
uv run python agent7.py C2            # MUST keep state/ from C1
uv run python agent7.py --reset D     # asyncio multi-source synthesis
uv run python agent7.py --reset E     # index papers/attention.md + three contributions
uv run python agent7.py --reset F1    # index every papers/*.md, report chunk count
uv run python agent7.py F2            # MUST keep state/ from F1 — CoT across the index
uv run python agent7.py G             # semantic: credit assignment (phrase not in files)
uv run python agent7.py H             # ReAct vs Chain-of-Thought intermediate reasoning
```

Or everything, with the required cleans around C1/C2 and F1/F2:

```bash
uv run python agent7.py --all
```

`--all` wipes between A, B, C-pair, D, E. **F1 → F2 → G → H share `state/`** so the index persists.

Expected iteration counts (passing = at most 2× these): A 3, B ~5, C1 4, C2 2, D 5, E 4, F1 7, F2 3, G 3, H 3.

## Custom Formula One RAG app (50+ chunks)

Corpus + manifest: `corpus/`, `papers/`, `CORPUS_MANIFEST.md`. After `build_index.py --reset` this machine indexed **51 chunks** (5 paper abstracts + 13 F1 files).

Five queries in `custom_queries.py`. Each must **hit with the index** and **fail without it**. FQ3 and FQ4 are semantic.

```bash
uv run python build_index.py --reset          # papers + corpus → 51 chunks
uv run python run_custom_queries.py --compare # without index, then with
```

| ID | Query (short) | Semantic |
|----|----------------|----------|
| FQ1 | 2003 neck-protection device (HANS) | no |
| FQ2 | When ground-effect tunnels returned | no |
| FQ3 | 2022 vertical oscillation / bouncing (porpoising) | **yes** |
| FQ4 | Which UK driver matched the all-time title record | **yes** |
| FQ5 | Undercut vs overcut | no |

## Perception grep gate

```bash
rg -i "web_search|fetch_url|index_document|search_knowledge|read_file|list_dir|create_file" prompts/perception.txt
# no matches — Perception SYSTEM does not name MCP tools
```

Tool names live in `prompts/decision.txt` and the MCP descriptors Action forwards.

## Safety nets (code, not prompt hope)

Session 6 carry-over: sticky-done, evidence-done, force-attach, position-based `attach_index`, incremental artifact ids, Action refuses artifact ids / `mem:` as `url`/`path` (not as integer `k`).

Session 7 additions:

- Vector-first `Memory.read`; keyword fallback only if FAISS is empty (not RRF hybrid)
- `index_document` / `search_knowledge` auto-picks when the current goal is clearly an index or retrieve goal
- Query F1 seeds one index goal per official paper, then a confirm-count goal
- Cross-paper questions seed retrieve → synthesise (no web search)
- Synthesis with attached RAG/fetch bytes is answered by Gemini, not `read_file`
- Empty index → `No relevant indexed chunks found.` (the without-index custom path)

## YouTube

Add the demo link here after recording A–H plus one with-index / without-index custom pair.

## Terminal output (clean state, this machine)

Noise from crawl4ai / FastMCP / httpx is omitted. Lines are the agent trace: `memory.read`, Perception goals, Decision, Action descriptor, FINAL.

### Query A — 3 iters

```
─── iter 1 ───
[memory.read]   0 hits (vector-first, keyword fallback)
[perception]    [open] Fetch https://en.wikipedia.org/wiki/Claude_Shannon
                [open] Extract birth date, death date, and three key contributions …
[decision]      TOOL_CALL: fetch_url({"url": "https://en.wikipedia.org/wiki/Claude_Shannon"})
[action]        → [artifact id=1, 256691 bytes]

─── iter 2 ───
[attach]        1 (256691 bytes)
[decision]      ANSWER: Claude Shannon was born on April 30, 1916, and died on February 24, 2001. …

─── iter 3 ───
[done] all 2 goals satisfied

FINAL: Claude Shannon was born on April 30, 1916, and died on February 24, 2001. His three key contributions to information theory include: 1) The development of information entropy as a measure of information content and uncertainty; 2) The establishment of the fundamental limits on data transmission and encoding (often referred to as the 'Magna Carta of the Information Age'); and 3) The introduction of the 'bit' as the basic unit of information, alongside the development of the sampling theorem (Nyquist–Shannon sampling theorem) which enabled the transition from analog to digital telecommunications.
```

### Query B — 5 iters

Search family-friendly Tokyo activities, check Saturday weather, pick the indoor option.

```
FINAL: Considering the weather forecast for Saturday, August 22, in Tokyo—which predicts windy conditions and a 65% chance of afternoon rain—I recommend visiting the Tokyo Water Science Museum. It is an excellent indoor, family-friendly activity that will allow you to enjoy your day regardless of the outdoor weather.
```

### Query C run 1 — 4 iters

```
[memory.remember]  classified as fact
[decision]      TOOL_CALL: create_file({"path": "reminders/1_May_2026_reminder", …})
[decision]      TOOL_CALL: create_file({"path": "reminders/15_May_2026_reminder", …})
[decision]      ANSWER: The reminders for 1 May 2026 and 15 May 2026 have been set.

FINAL: The reminders for 1 May 2026 and 15 May 2026 have been set.
```

### Query C run 2 — 2 iters (no `--reset`; 0 tools)

```
[memory.read]   3 hits (vector-first, keyword fallback)
[perception]    [open] Answer the user using the fact from memory
[decision]      ANSWER: My mom's birthday is on 15 May 2026.

FINAL: My mom's birthday is on 15 May 2026.
```

### Query D — 5 iters

```
[decision]      TOOL_CALL: web_search({"query": "Python asyncio best practices", "max_results": 5})
[decision]      TOOL_CALL: fetch_url({"url": "https://discuss.python.org/t/asyncio-best-practices/12576"})
[decision]      TOOL_CALL: fetch_url({"url": "https://docs.python.org/3/library/asyncio-dev.html"})
[attach]        1 (6503 bytes)
[attach]        2 (26168 bytes)
[decision]      ANSWER: … numbered list of agreed-upon advice …

FINAL: … 1. Avoid blocking the event loop … Use 'loop.run_in_executor()' … 2. Use Debug Mode during development … 3. Manage asynchronous generators carefully … 'contextlib.aclosing()' …
```

### Query E — 4 iters

```
[decision]      TOOL_CALL: index_document({"path": "papers/attention.md"})
[action]        → indexed papers/attention.md → 3 chunks
[decision]      TOOL_CALL: search_knowledge({…Transformer…, "k": 5})
[action]        → search_knowledge returned 3 chunks
[attach]        1 (3756 bytes)

FINAL: The three key contributions of the Transformer architecture are: 1) It relies solely on attention mechanisms, completely dispensing with recurrence and convolutions. 2) It is significantly more parallelizable than previous models, leading to faster training times. 3) It achieves superior translation quality, establishing new state-of-the-art results on WMT 2014 English-to-German and English-to-French translation tasks.
```

### Query F run 1 — 7 iters (`--reset F1`)

```
[decision]      TOOL_CALL: index_document({"path": "papers/attention.md"})
[decision]      TOOL_CALL: index_document({"path": "papers/cot.md"})
[decision]      TOOL_CALL: index_document({"path": "papers/react.md"})
[decision]      TOOL_CALL: index_document({"path": "papers/dpo.md"})
[decision]      TOOL_CALL: index_document({"path": "papers/lora.md"})
[decision]      ANSWER: The total indexed chunk count is 5.

FINAL: The total indexed chunk count is 5.
```

(Each official paper abstracts to one FAISS chunk after chrome-strip. The 51-chunk total is papers + F1 corpus via `build_index.py`.)

### Query F run 2 — 3 iters (`F2`, no reset)

```
[decision]      TOOL_CALL: search_knowledge({…chain-of-thought reasoning…, "k": 5})
[action]        → search_knowledge returned 5 chunks

FINAL: Based on the retrieved papers, chain-of-thought (CoT) reasoning is a technique where large language models are prompted to generate a series of intermediate reasoning steps. … the 'ReAct' paper indicates that CoT reasoning is often studied alongside action plan generation …
```

### Query G — 3 iters (semantic; phrase **not** in the files)

```
[decision]      TOOL_CALL: search_knowledge({…credit assignment problem…, "k": 5})

FINAL: Across the indexed papers, credit assignment is handled through different paradigms: Direct Preference Optimization (DPO) simplifies the reward-based credit assignment typically found in RLHF by reformulating the objective to optimize the language model directly against human preferences … ReAct addresses credit assignment in interactive tasks by interleaving reasoning traces with actions, allowing the model to attribute outcomes to specific steps in its thought-action sequence.
```

### Query H — 3 iters

```
[decision]      TOOL_CALL: search_knowledge({…ReAct…Chain-of-Thought…intermediate reasoning…, "k": 5})

FINAL: The Chain-of-Thought (CoT) paper focuses on generating a series of intermediate reasoning steps … In contrast, the ReAct paper treats intermediate reasoning as part of a broader, integrated framework that synergizes 'reasoning' with 'acting.' While CoT treats reasoning as a self-contained process for eliciting logic, ReAct explicitly combines reasoning traces with the generation of action plans …
```

### Custom F1 queries (`run_custom_queries.py --compare`)

Index rebuilt to **51 chunks**.

**Without index** (all five): `search_knowledge` returned 0 chunks →

```
FINAL: No relevant indexed chunks found. The knowledge index is empty or does not cover this question.
```

**With index:**

| ID | Result |
|----|--------|
| FQ1 | **HANS** device, mandatory 2003, tethers helmet to shoulders |
| FQ2 | Ground-effect tunnels returned in the **2022** aero reset (Lotus 79 / 1983 ban as history) |
| FQ3 | Semantic: **porpoising** — violent vertical oscillation when 2022 underfloor load peaked |
| FQ4 | Semantic: **Lewis Hamilton** (United Kingdom), seven titles, tied with Schumacher |
| FQ5 | **Undercut** = pit earlier on fresh tyres; **overcut** = stay out longer in clean air and pit later |
