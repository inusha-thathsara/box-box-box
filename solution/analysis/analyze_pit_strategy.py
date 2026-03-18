#!/usr/bin/env python3
"""Analyze pit strategy success in historical races."""

import json
from pathlib import Path
from collections import defaultdict

# Load historical races
races = []
for file_path in sorted(Path("..").glob("data/historical_races/races_*.json")):
    try:
        with open(file_path) as f:
            races.extend(json.load(f))
    except:
        pass

print(f"Loaded {len(races)} races")
print("=" * 80)

# Analyze pit strategies
early_pit_wins = 0  # pit_lap <= 10
mid_pit_wins = 0    # 11 <= pit_lap <= 18
late_pit_wins = 0   # pit_lap >= 19
early_pit_total = 0
mid_pit_total = 0
late_pit_total = 0

for race in races:
    if not race.get("finishing_positions"):
        continue
    
    winner = race["finishing_positions"][0]
    
    # Find winner's pit strategy
    for pos in range(1, 21):
        strategy = race["strategies"][f"pos{pos}"]
        if strategy["driver_id"] == winner:
            pit_stops = strategy.get("pit_stops", [])
            if pit_stops:
                first_pit_lap = min(int(s["lap"]) for s in pit_stops)
                if first_pit_lap <= 10:
                    early_pit_wins += 1
                    early_pit_total += 1
                elif first_pit_lap <= 18:
                    mid_pit_wins += 1
                    mid_pit_total += 1
                else:
                    late_pit_wins += 1
                    late_pit_total += 1
            break

print("WINNER PIT STRATEGIES:")
print("-" * 80)
print(f"Early pit (lap ≤10): {early_pit_wins} wins")
print(f"Mid pit (lap 11-18): {mid_pit_wins} wins")
print(f"Late pit (lap ≥19):  {late_pit_wins} wins")
print(f"Total winners with pits: {early_pit_wins + mid_pit_wins + late_pit_wins}")

# Analyze top 5 finishers
print("\n" + "=" * 80)
print("TOP 5 FINISHERS BY PIT STRATEGY:")
print("-" * 80)

top5_early = 0
top5_mid = 0
top5_late = 0

for race in races:
    if not race.get("finishing_positions"):
        continue
    
    for rank in range(min(5, len(race["finishing_positions"]))):
        driver = race["finishing_positions"][rank]
        
        for pos in range(1, 21):
            strategy = race["strategies"][f"pos{pos}"]
            if strategy["driver_id"] == driver:
                pit_stops = strategy.get("pit_stops", [])
                if pit_stops:
                    first_pit_lap = min(int(s["lap"]) for s in pit_stops)
                    if first_pit_lap <= 10:
                        top5_early += 1
                    elif first_pit_lap <= 18:
                        top5_mid += 1
                    else:
                        top5_late += 1
                break

print(f"Early pit (lap ≤10): {top5_early} top-5 finishes")
print(f"Mid pit (lap 11-18): {top5_mid} top-5 finishes")
print(f"Late pit (lap ≥19):  {top5_late} top-5 finishes")

print("\n" + "=" * 80)
print("INSIGHT: Do late pits hurt or help?")
print(f"  Late pit strategy win rate among winners: {late_pit_wins}/{late_pit_total}")
print(f"  Late pit in top 5: {top5_late} (should be ≥1 if valuable)")
