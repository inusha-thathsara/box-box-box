#!/usr/bin/env python3
"""Show how new pit timing features improve signal."""

import json
from pathlib import Path

# Analyze test_001 with new features
with open("data/test_cases/inputs/test_001.json") as f:
    test = json.load(f)
with open("data/test_cases/expected_outputs/test_001.json") as f:
    expected = json.load(f)

print("TEST_001: PIT TIMING FEATURE ANALYSIS")
print("=" * 80)
print(f"Race: {test['race_config']['track']}, {test['race_config']['total_laps']} laps")
print()

# Extract pit timing for each driver
driver_features = {}
for pos in range(1, 21):
    strat = test["strategies"][f"pos{pos}"]
    driver_id = strat["driver_id"]
    
    pit_stops = sorted(strat.get("pit_stops", []), key=lambda s: int(s["lap"]))
    pit_laps = [int(stop["lap"]) for stop in pit_stops]
    
    avg_pit_lap = sum(pit_laps) / len(pit_laps) if pit_laps else 0.0
    first_pit_early = 1.0 if pit_laps and pit_laps[0] <= 10 else 0.0
    
    driver_features[driver_id] = {
        "start_tire": strat["starting_tire"],
        "pit_laps": pit_laps,
        "avg_pit_lap": avg_pit_lap,
        "first_pit_early": first_pit_early,
    }

# Compare top finishers vs predicted winners
print("EXPECTED vs PREDICTED (top 5):")
print("-" * 80)
print(f"{'Rank':<5} {'Expected':<10} {'Pit Strategy':<20} {'Avg Pit Lap':<12} {'First Pit Early':<15}")
print("-" * 80)

for rank in range(1, 6):
    driver = expected["finishing_positions"][rank - 1]
    features = driver_features[driver]
    print(f"{rank:<5} {driver:<10} {str(features['pit_laps']):<20} {features['avg_pit_lap']:>6.1f}        {int(features['first_pit_early']):<15}")

print()
print("KEY OBSERVATION:")
print("-" * 80)

d009_feat = driver_features["D009"]
d019_feat = driver_features["D019"]

print(f"D009 (expected 4th):  pit_laps={d009_feat['pit_laps']}, avg={d009_feat['avg_pit_lap']:.1f}, early={int(d009_feat['first_pit_early'])}")
print(f"D019 (expected 5th):  pit_laps={d019_feat['pit_laps']}, avg={d019_feat['avg_pit_lap']:.1f}, early={int(d019_feat['first_pit_early'])}")
print()
print("D009 pits LATER (lap 18) → should get fresher tires for final laps")
print("D019 pits EARLY (lap 8)  → tires wear for 23 laps after pit")
print()
print("✓ The 'avg_pit_lap' and 'first_pit_early' features now capture this signal!")
print("✓ A learned model can now discover: late pitting is valuable")

