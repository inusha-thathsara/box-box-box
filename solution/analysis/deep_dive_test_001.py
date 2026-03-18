#!/usr/bin/env python3
"""Deep dive into a specific test case failure."""

import json
import sys
from pathlib import Path

def run_race_simulator(test_json):
    import subprocess
    result = subprocess.run(
        ["python", "solution/race_simulator.py"],
        input=json.dumps(test_json),
        capture_output=True,
        text=True,
        cwd="."
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except:
        return None

# Analyze test_001
with open("data/test_cases/inputs/test_001.json") as f:
    test = json.load(f)
with open("data/test_cases/expected_outputs/test_001.json") as f:
    expected = json.load(f)

pred = run_race_simulator(test)

print("TEST_001 DEEP DIVE")
print("=" * 80)
print(f"Race: {test['race_config']['track']}, {test['race_config']['total_laps']} laps, {test['race_config']['track_temp']}°C")
print()

# Show each driver and their strategy
strategies = {}
for pos in range(1, 21):
    strategy = test["strategies"][f"pos{pos}"]
    driver_id = strategy["driver_id"]
    pit_count = len(strategy.get("pit_stops", []))
    pit_details = []
    for stop in strategy.get("pit_stops", []):
        pit_details.append(f"L{stop['lap']}: {stop['from_tire']}→{stop['to_tire']}")
    pit_str = ", ".join(pit_details) if pit_details else "No pits"
    strategies[driver_id] = {
        "start_tire": strategy["starting_tire"],
        "pit_count": pit_count,
        "pit_details": pit_str,
    }

print("Driver strategies (sorted by expected finish position):")
print("-" * 80)
for rank, driver in enumerate(expected["finishing_positions"], 1):
    strat = strategies[driver]
    print(f"P{rank:2d}: {driver} | Start: {strat['start_tire']:6s} | Pits: {strat['pit_count']} | {strat['pit_details']}")

print()
print("COMPARISON:")
print("-" * 80)
for rank in range(1, 11):
    exp_driver = expected["finishing_positions"][rank - 1]
    pred_driver = pred["finishing_positions"][rank - 1]
    exp_strat = strategies[exp_driver]
    pred_strat = strategies[pred_driver]
    
    match = "✓" if exp_driver == pred_driver else "✗"
    print(f"Pos {rank:2d}: {match} Expected {exp_driver} | Predicted {pred_driver}")
    print(f"         Exp: {exp_strat['start_tire']} + {exp_strat['pit_count']} stops | Pred: {pred_strat['start_tire']} + {pred_strat['pit_count']} stops")

# Find where D009 is predicted
d009_pred_pos = pred["finishing_positions"].index("D009")
print()
print(f"D009: Expected 4th position, Predicted {d009_pred_pos + 1}th position")
print(f"      Start: {strategies['D009']['start_tire']}, Pits: {strategies['D009']['pit_count']}")
print(f"      Details: {strategies['D009']['pit_details']}")

# Compare lap times estimate
print()
print("ANALYZING: Why is D009 predicted so low?")
print("-" * 80)

# Extract pit info
def get_pit_schedule(strategy):
    return {s["lap"]: s["to_tire"] for s in strategy.get("pit_stops", [])}

def estimate_simple_time(trace_config, strategy):
    """Simple lap time estimate to debug."""
    total_laps = trace_config["total_laps"]
    base_lap = trace_config["base_lap_time"]
    
    # Very rough: soft = -0.5, medium = 0, hard = +0.5
    tire_offsets = {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.5}
    
    current_tire = strategy["starting_tire"]
    total_time = 0.0
    pit_schedule = get_pit_schedule(strategy)
    
    for lap in range(1, total_laps + 1):
        lap_time = base_lap + tire_offsets[current_tire]
        if lap in pit_schedule:
            total_time += trace_config["pit_lane_time"]
            current_tire = pit_schedule[lap]
        total_time += lap_time
    
    return total_time

# Compare D004 (expected 3rd) vs D009 (expected 4th)
d004_strat = test["strategies"]["pos4"]  # D004 is in pos4 input
d009_strat = test["strategies"]["pos9"]  # D009 is in pos9 input

# Find actual position inputs
d004_pos = None
d009_pos = None
for pos in range(1, 21):
    if test["strategies"][f"pos{pos}"]["driver_id"] == "D004":
        d004_pos = pos
    if test["strategies"][f"pos{pos}"]["driver_id"] == "D009":
        d009_pos = pos

print(f"D004 input position: pos{d004_pos}, Start: {test['strategies'][f'pos{d004_pos}']['starting_tire']}, Pits: {len(test['strategies'][f'pos{d004_pos}'].get('pit_stops', []))}")
print(f"D009 input position: pos{d009_pos}, Start: {test['strategies'][f'pos{d009_pos}']['starting_tire']}, Pits: {len(test['strategies'][f'pos{d009_pos}'].get('pit_stops', []))}")

# Rough time estimate
d004_rough = estimate_simple_time(test["race_config"], test["strategies"][f"pos{d004_pos}"])
d009_rough = estimate_simple_time(test["race_config"], test["strategies"][f"pos{d009_pos}"])

print(f"\nRough time estimate: D004={d004_rough:.1f}s, D009={d009_rough:.1f}s")
if d009_rough < d004_rough:
    print(f"  → D009 should be FASTER by {d004_rough - d009_rough:.1f}s, but is ranked lower!")

