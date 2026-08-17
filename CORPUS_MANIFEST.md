# Corpus manifest (Assignment 7)

Target: **50+ indexed chunks** — **51 chunks** after `uv run python build_index.py --reset`.

Chunker: 400 words, 80-word overlap (Session 7 defaults). Embeddings: Gateway V7 `POST /v1/embed` → Ollama `nomic-embed-text` (768-d), FAISS `IndexFlatIP` with L2-normalized vectors.

Paper files are the official Session 7 arXiv abstracts. Indexing strips arXiv HTML chrome (nav, submission history) so the Abstract is what FAISS stores. The phrase **“credit assignment” does not appear** in any paper file (Query G is synonym / vector recall).

## Professor papers (`papers/`) — queries E–H

| File | Paper | Chunks after chrome-strip |
|------|--------|---------------------------|
| `attention.md` | Vaswani et al., Attention Is All You Need | 1 |
| `cot.md` | Wei et al., Chain-of-Thought Prompting | 1 |
| `react.md` | Yao et al., ReAct | 1 |
| `dpo.md` | Rafailov et al., Direct Preference Optimization | 1 |
| `lora.md` | Hu et al., LoRA | 1 |

Query F1 indexes these five files only and reports **5 chunks**.

## Custom corpus (`corpus/`) — Formula One RAG app

| File | Topic | Chunks |
|------|--------|--------|
| `f1_history.md` | championships, 2022 ground effect, HANS, halo | 4 |
| `f1_hybrid_power_units.md` | V6 turbo-hybrid, 2022 tunnels, porpoising | 4 |
| `f1_safety_innovations.md` | HANS (2003), halo, barriers | 4 |
| `f1_legendary_drivers.md` | Schumacher, Hamilton (UK, seven titles), Verstappen | 4 |
| `f1_circuits.md` | Silverstone, Monaco, Spa, … | 3 |
| `f1_strategy_and_pit_stops.md` | undercut vs overcut | 4 |
| `f1_tyres.md` | compounds, undercut interaction | 2 |
| `f1_constructors.md` | team structure | 2 |
| `f1_regulations.md` | technical directives | 2 |
| `f1_overtaking.md` | DRS, tyre delta | 1 |
| `f1_calendar.md` | grands prix | 2 |
| `f1_seasons.md` | year-by-year notes | 12 |
| `f1_flags.md` | yellow / red / blue / VSC | 2 |

## Totals

- Raw documents: 18 (5 papers + 13 F1 files)
- Indexed chunks: **51**
- Gateway: `llm_gatewayV7/` on port **8107** (professor ZIP)

## Custom queries

See `custom_queries.py`. FQ3 (porpoising / violent vertical oscillation) and FQ4 (Hamilton / UK titles) are semantic — the query does not name the answer string. Run `uv run python run_custom_queries.py --compare`.
