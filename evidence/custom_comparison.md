# Custom Formula One RAG comparison

## Without index

### FQ1

**Query:** What safety device became mandatory in 2003 to protect drivers' necks in frontal impacts?

**Result:** FAIL (expected)

**Answer:** FAIL — no indexed Formula One evidence was retrieved.

### FQ2

**Query:** When did Formula One reintroduce ground-effect tunnels under the floor?

**Result:** FAIL (expected)

**Answer:** FAIL — no indexed Formula One evidence was retrieved.

### FQ3

**Query:** In the racing collection, what animal-like term describes the rapid up-and-down motion of the new underfloor cars?

**Result:** FAIL (expected)

**Answer:** FAIL — no indexed Formula One evidence was retrieved.

### FQ4

**Query:** Which British competitor equalled the championship benchmark previously held alone by the dominant German Ferrari driver, and how many titles did each hold?

**Result:** FAIL (expected)

**Answer:** FAIL — no indexed Formula One evidence was retrieved.

### FQ5

**Query:** Explain the difference between an undercut and an overcut during a Grand Prix.

**Result:** FAIL (expected)

**Answer:** FAIL — no indexed Formula One evidence was retrieved.

## With index

### FQ1

**Query:** What safety device became mandatory in 2003 to protect drivers' necks in frontal impacts?

**Result:** PASS

**Answer:** The Head and Neck Support (HANS) device became mandatory in 2003 to protect drivers' necks in frontal impacts.
Sources: corpus:f1_safety_innovations.md#hans-device, corpus:f1_safety_innovations.md#halo, corpus:f1_safety_innovations.md#black-and-orange-flag, corpus:f1_regulations.md#safety-car-vsc-and-red-flags, corpus:f1_safety_innovations.md#overview

### FQ2

**Query:** When did Formula One reintroduce ground-effect tunnels under the floor?

**Result:** PASS

**Answer:** Formula One reintroduced ground-effect tunnels under the floor in the 2022 regulations, aiming to improve close racing by reducing the wake problem that hinders overtaking.
Sources: corpus:f1_hybrid_power_units.md#2022-aero-reset-and-porpoising, corpus:f1_history.md#overview, corpus:f1_overtaking.md#overview, corpus:f1_overtaking.md#2022-2024-following-behavior, corpus:f1_seasons.md#1978

### FQ3

**Query:** In the racing collection, what animal-like term describes the rapid up-and-down motion of the new underfloor cars?

**Result:** PASS

**Answer:** The animal-like term that describes the rapid up-and-down motion of the new underfloor cars is **porpoising**.
Sources: corpus:f1_hybrid_power_units.md#2022-aero-reset-and-porpoising, corpus:f1_strategy_and_pit_stops.md#overview, corpus:f1_overtaking.md#overview, corpus:f1_circuits.md#autodromo-nazionale-monza-italy, corpus:f1_overtaking.md#2022-2024-following-behavior

### FQ4

**Query:** Which British competitor equalled the championship benchmark previously held alone by the dominant German Ferrari driver, and how many titles did each hold?

**Result:** PASS

**Answer:** Lewis Hamilton of the United Kingdom equalled the championship benchmark previously held alone by Michael Schumacher of Germany, with each holding seven titles.
Sources: corpus:f1_legendary_drivers.md#michael-schumacher-germany, corpus:f1_legendary_drivers.md#statistical-notes, corpus:f1_legendary_drivers.md#lewis-hamilton-united-kingdom, corpus:f1_history.md#overview, corpus:f1_seasons.md#2000

### FQ5

**Query:** Explain the difference between an undercut and an overcut during a Grand Prix.

**Result:** PASS

**Answer:** An undercut involves pitting before a rival to gain an advantage with fresh tires, while an overcut involves staying out longer to use clean air and warm tires, potentially allowing a later pit stop to regain a position.
Sources: corpus:f1_strategy_and_pit_stops.md#overview, corpus:f1_strategy_and_pit_stops.md#undercut-and-overcut, corpus:f1_overtaking.md#tyre-offset-as-an-overtaking-tool, corpus:f1_overtaking.md#overview, corpus:f1_circuits.md#suzuka-japan
