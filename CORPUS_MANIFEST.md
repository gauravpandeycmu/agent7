# Corpus manifest — Formula One RAG application

The custom application contains **131 independently indexed knowledge items** across 13 Markdown documents. This exceeds the assignment minimum of 50 items without counting the five course papers used by base Queries E–H.

`build_index.py` turns each document overview and each meaningful `##` section into one item. Every item receives its own descriptor, source identifier, embedding, Memory record, and FAISS entry. Boilerplate sections whose headings begin with `Supplement` are ignored.

## Formula One corpus

| File | Subject | Indexed items |
|---|---|---:|
| `f1_calendar.md` | Calendar structure and events | 1 |
| `f1_circuits.md` | Major circuits | 6 |
| `f1_constructors.md` | Constructor organization | 2 |
| `f1_flags.md` | Race-control flags | 1 |
| `f1_history.md` | Championship history | 1 |
| `f1_hybrid_power_units.md` | Hybrid engines, 2022 floor rules, porpoising | 7 |
| `f1_legendary_drivers.md` | Schumacher, Hamilton, and other champions | 6 |
| `f1_overtaking.md` | DRS, tyre offset, and following behavior | 6 |
| `f1_regulations.md` | Sporting and technical rules | 7 |
| `f1_safety_innovations.md` | HANS, halo, barriers, and procedures | 8 |
| `f1_seasons.md` | Overview plus 75 season records | 76 |
| `f1_strategy_and_pit_stops.md` | Undercut, overcut, and pit strategy | 8 |
| `f1_tyres.md` | Compounds and tyre behavior | 2 |
| **Total** | **13 documents** | **131 items** |

Reproduce the count without invoking a model:

```bash
uv run python build_index.py --count-only
# 131
```

Build the index and run both sides of the comparison:

```bash
uv run python run_custom_queries.py --compare \
  --output evidence/custom_comparison.md
```

## Semantic-query proof

| Query | Wording deliberately absent from the corpus | Retrieved answer concept |
|---|---|---|
| FQ3 | `rapid up-and-down motion` | `porpoising` from `f1_hybrid_power_units.md` |
| FQ4 | `British competitor` | Lewis Hamilton and Michael Schumacher, seven titles each, from `f1_legendary_drivers.md` |

`verify_assignment.py` checks that both probe phrases are absent from every Formula One Markdown file. The expected answer terms are not present in the queries, so an empty index cannot answer either case from query text alone.

## Course papers used by base Queries E–H

The five files under `papers/` are the supplied Session 7 paper texts:

| File | Paper | Query F1 chunks |
|---|---|---:|
| `attention.md` | Attention Is All You Need | 3 |
| `cot.md` | Chain-of-Thought Prompting | 3 |
| `react.md` | ReAct | 3 |
| `dpo.md` | Direct Preference Optimization | 3 |
| `lora.md` | LoRA | 3 |
| **Total** | **5 supplied papers** | **15 chunks** |

These use the assignment defaults of 400 words per chunk with an 80-word overlap. They are kept separate from the 131-item custom-corpus count.
