#!/usr/bin/env python3
"""Analyze tire+pit combinations for winners."""

import json
from pathlib import Path
from collections import Counter

races = []
for file_path in sorted(Path("..").glob("data/historical_races/races_*.json")):
    try:
        with open(file_path) as f:
            races.extend(json.load(f))
    except:
        pass

# Analyze winner tire+pit combos
winner_combos = Counter()

for race in races:
    if not race.get("finishing_positions"):
        continue
    
    winner = race["finishing_positions"][0]
    
    for pos in range(1, 21):
        strategy = race["strategies"][f"pos{pos}"]
        if strategy["driver_id"] == winner:
            start_tire = strategy["starting_tire"]
            pit_stops = strategy.get("pit_stops", [])
            
            if pit_stops:
                first_pit_lap = min(int(s["lap"]) for s in pit_stops)
                pit_to_tire = next((s["to_tire"] for s in pit_stops if int(s["lap"]) == first_pit_lap), "?")
                
                if first_pit_lap <= 10:
                    pit_category = "early"
                elif first_pit_lap <= 18:
                    pit_category = "mid"
                else:
                    pit_category = "late"
                
                combo = f"{start_tire} → {pit_to_tire} (pit_lap {pit_category})"
                winner_combos[combo] += 1
            break

print("TOP WINNING TIRE COMBINATIONS:")
print("=" * 80)
for combo, count in winner_combos.most_common(15):
    pct = 100.0 * count / 30000
    print(f"{combo:<35} {count:>5} wins ({pct:>5.1f}%)")
