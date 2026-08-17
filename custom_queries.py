"""Five Formula One questions used by the custom RAG demonstration."""

CUSTOM_QUERIES: list[dict[str, object]] = [
    {
        "id": "FQ1",
        "query": (
            "What safety device became mandatory in 2003 to protect drivers' "
            "necks in frontal impacts?"
        ),
        "expected_terms": ["HANS", "Head and Neck Support", "2003"],
        "semantic": False,
    },
    {
        "id": "FQ2",
        "query": (
            "When did Formula One reintroduce ground-effect tunnels under "
            "the floor?"
        ),
        "expected_terms": ["2022"],
        "semantic": False,
    },
    {
        "id": "FQ3",
        "query": (
            "In the racing collection, what animal-like term describes the "
            "rapid up-and-down motion of the new underfloor cars?"
        ),
        "expected_terms": ["porpoising"],
        "semantic": True,
        "semantic_probe": "rapid up-and-down motion",
        "target_source": "f1_hybrid_power_units.md",
    },
    {
        "id": "FQ4",
        "query": (
            "Which British competitor equalled the championship benchmark "
            "previously held alone by the dominant German Ferrari driver, "
            "and how many titles did each hold?"
        ),
        "expected_terms": ["Lewis Hamilton", "seven", "Michael Schumacher"],
        "semantic": True,
        "semantic_probe": "British competitor",
        "target_source": "f1_legendary_drivers.md",
    },
    {
        "id": "FQ5",
        "query": (
            "Explain the difference between an undercut and an overcut during "
            "a Grand Prix."
        ),
        "expected_terms": ["undercut", "pitting before", "overcut", "staying out longer"],
        "semantic": False,
    },
]
