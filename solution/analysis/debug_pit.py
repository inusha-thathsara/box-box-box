#!/usr/bin/env python3
"""Debug pit compensation for test_001."""

import json

with open("data/test_cases/inputs/test_001.json") as f:
    test = json.load(f)

print("PIT COMPENSATION DEBUG FOR TEST_001")
print("=" * 80)

total_laps = test["race_config"]["total_laps"]

for pos in [4, 5]:  # D009 is pos9, D019 is pos19
    strategy = test["strategies"][f"pos{pos}"]
    driver_id = strategy["driver_id"]
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))
    
    if pit_stops:
        first_pit_lap = int(pit_stops[0]["lap"])
        
        if first_pit_lap >= 16:
            bonus = -1.5 * (first_pit_lap - 15) / float(total_laps)
            print(f"{driver_id} (pit_lap {first_pit_lap}): LATE PIT BONUS = {bonus:.4f}s")
        elif first_pit_lap <= 10:
            penalty = 0.8 * (10 - first_pit_lap) / 10.0
            print(f"{driver_id} (pit_lap {first_pit_lap}): EARLY PIT PENALTY = +{penalty:.4f}s")

print()
print("ISSUE: Bonus is too small! Need at least 1s difference to flip ranking.")
print("Let me try stronger values...")

for pos in [4, 5]:
    strategy = test["strategies"][f"pos{pos}"]
    driver_id = strategy["driver_id"]
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))
    
    if pit_stops:
        first_pit_lap = int(pit_stops[0]["lap"])
        
        if first_pit_lap >= 16:
            # Stronger: -3.0 instead of -1.5
            bonus = -3.0 * (first_pit_lap - 15) / float(total_laps)
            print(f"{driver_id} (pit_lap {first_pit_lap}): STRONGER BONUS = {bonus:.4f}s")
        elif first_pit_lap <= 10:
            # Stronger: 1.5 instead of 0.8
            penalty = 1.5 * (10 - first_pit_lap) / 10.0
            print(f"{driver_id} (pit_lap {first_pit_lap}): STRONGER PENALTY = +{penalty:.4f}s")

print()
# Find D009 and D019 actual positions
for pos in range(1, 21):
    driver = test["strategies"][f"pos{pos}"]["driver_id"]
    pit_stops = test["strategies"][f"pos{pos}"].get("pit_stops", [])
    if pit_stops:
        first_lap = int(pit_stops[0]["lap"])
        if driver in ["D009", "D019"]:
            print(f"Pos {pos}: {driver} pits at lap {first_lap}")
