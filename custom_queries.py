"""Five custom Assignment 7 queries — Formula One RAG corpus.

Each query must pass with index populated and fail/degrade without it.
At least two use semantic recall (query words absent from chunk keywords).
"""

CUSTOM_QUERIES: list[dict[str, str | bool]] = [
    {
        "id": "FQ1",
        "query": "What safety device became mandatory in 2003 to protect drivers' necks in frontal impacts?",
        "needs_index": True,
        "semantic": False,
        "note": "Keyword hit: HANS in f1_safety_innovations.md",
    },
    {
        "id": "FQ2",
        "query": "When did Formula One reintroduce ground effect tunnels under the floor?",
        "needs_index": True,
        "semantic": False,
        "note": "2022 regulations in f1_hybrid_power_units.md / f1_history.md",
    },
    {
        "id": "FQ3",
        "query": "What violent vertical oscillation did 2022 ground-effect cars suffer when aerodynamic load peaked?",
        "needs_index": True,
        "semantic": True,
        "note": "Semantic: porpoising — query may not say 'porpoising'",
    },
    {
        "id": "FQ4",
        "query": "Which UK driver matched the all-time record for most drivers' titles?",
        "needs_index": True,
        "semantic": True,
        "note": "Semantic: Lewis Hamilton — query does not say Hamilton or seven",
    },
    {
        "id": "FQ5",
        "query": "Explain the difference between an undercut and an overcut during a Grand Prix.",
        "needs_index": True,
        "semantic": False,
        "note": "f1_strategy_and_pit_stops.md — fails without index",
    },
]
