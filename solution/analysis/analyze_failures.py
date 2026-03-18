#!/usr/bin/env python3
"""Analyze which test cases pass/fail and identify patterns."""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

def run_race_simulator(test_json):
    """Call race_simulator.py on a test case."""
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

def main():
    test_dir = Path("data/test_cases/inputs")
    expected_dir = Path("data/test_cases/expected_outputs")
    
    test_files = sorted(test_dir.glob("test_*.json"))
    
    passing = []
    failing = []
    
    print(f"Analyzing {len(test_files)} test cases...")
    print("-" * 80)
    
    for test_file in test_files:
        expected_file = expected_dir / test_file.name
        
        with open(test_file) as f:
            test_case = json.load(f)
        with open(expected_file) as f:
            expected = json.load(f)
        
        pred = run_race_simulator(test_case)
        if pred is None:
            print(f"ERROR running simulator on {test_file.name}")
            failing.append((test_file.name, test_case, expected, None))
            continue
        
        is_correct = pred["finishing_positions"] == expected["finishing_positions"]
        
        if is_correct:
            passing.append((test_file.name, test_case, expected, pred))
        else:
            failing.append((test_file.name, test_case, expected, pred))
    
    print(f"\n✓ PASSING: {len(passing)}/{len(test_files)}")
    print(f"✗ FAILING: {len(failing)}/{len(test_files)}")
    print("=" * 80)
    
    # Analyze passing cases
    if passing:
        print("\n[PASSING CASES ANALYSIS]")
        temp_dist = defaultdict(int)
        lap_dist = defaultdict(int)
        pit_dist = defaultdict(int)
        compound_starts = defaultdict(int)
        
        for name, test, exp, pred in passing:
            rc = test["race_config"]
            temp_dist[rc["track_temp"]] += 1
            lap_dist[rc["total_laps"]] += 1
            
            pit_count = sum(len(test["strategies"][f"pos{i}"].get("pit_stops", [])) for i in range(1, 21)) / 20.0
            pit_bucket = "low" if pit_count < 1.5 else "high"
            pit_dist[pit_bucket] += 1
            
            for i in range(1, 21):
                strat = test["strategies"][f"pos{i}"]
                compound_starts[strat["starting_tire"]] += 1
        
        print(f"Track temps: {dict(temp_dist)}")
        print(f"Total laps: {dict(lap_dist)}")
        print(f"Pit strategy: {dict(pit_dist)}")
        print(f"Starting compound distribution: {dict(compound_starts)}")
    
    # Analyze failing cases - driver ordering
    print("\n[FAILING CASES - MOST COMMON SWAP PAIRS]")
    swap_pairs = Counter()
    correct_pairs = Counter()
    
    for name, test, exp, pred in failing:
        expected_order = exp["finishing_positions"]
        predicted_order = pred["finishing_positions"]
        
        # Find swapped adjacent pairs
        for idx in range(len(expected_order) - 1):
            exp_a, exp_b = expected_order[idx], expected_order[idx + 1]
            
            # Find positions in prediction
            try:
                pred_a_idx = predicted_order.index(exp_a)
                pred_b_idx = predicted_order.index(exp_b)
            except ValueError:
                continue
            
            if abs(pred_a_idx - pred_b_idx) == 1:
                if pred_b_idx < pred_a_idx:
                    pair = tuple(sorted([exp_a, exp_b]))
                    swap_pairs[pair] += 1
            else:
                pair = tuple(sorted([exp_a, exp_b]))
                correct_pairs[pair] += 1
    
    print("Top 15 swapped pairs (pred has them reversed):")
    for pair, count in swap_pairs.most_common(15):
        print(f"  {pair[0]} ↔ {pair[1]}: {count} times")
    
    # Analyze failing by position
    print("\n[FAILING CASES - BY POSITION]")
    position_errors = defaultdict(int)
    position_totals = defaultdict(int)
    
    for name, test, exp, pred in failing:
        expected_order = exp["finishing_positions"]
        predicted_order = pred["finishing_positions"]
        
        for pos in range(len(expected_order)):
            exp_driver = expected_order[pos]
            pred_pos = predicted_order.index(exp_driver) if exp_driver in predicted_order else -1
            
            if pred_pos != pos:
                position_errors[pos] += 1
            position_totals[pos] += 1
    
    print("Positions with most errors (position_idx: error_count/total):")
    for pos in range(20):
        errors = position_errors[pos]
        total = position_totals[pos]
        pct = 100.0 * errors / total if total > 0 else 0
        if errors > 0:
            print(f"  Position {pos+1:2d}: {errors:2d}/{total} errors ({pct:5.1f}%)")
    
    # Analyze failing by driver
    print("\n[FAILING CASES - BY DRIVER]")
    driver_errors = defaultdict(int)
    driver_totals = defaultdict(int)
    
    for name, test, exp, pred in failing:
        expected_order = exp["finishing_positions"]
        predicted_order = pred["finishing_positions"]
        
        for driver in expected_order:
            exp_pos = expected_order.index(driver)
            pred_pos = predicted_order.index(driver)
            
            if exp_pos != pred_pos:
                driver_errors[driver] += 1
            driver_totals[driver] += 1
    
    print("Drivers with most misplacements:")
    driver_error_rate = [
        (driver, driver_errors[driver], driver_totals[driver])
        for driver in sorted(driver_totals.keys())
    ]
    driver_error_rate.sort(key=lambda x: -x[1])
    
    for driver, errors, total in driver_error_rate[:10]:
        pct = 100.0 * errors / total if total > 0 else 0
        print(f"  {driver}: {errors}/{total} errors ({pct:.1f}%)")
    
    # Analyze failing by race config
    print("\n[FAILING CASES - BY RACE CONFIG]")
    config_errors = defaultdict(int)
    config_totals = defaultdict(int)
    
    for name, test, exp, pred in failing:
        track = test["race_config"]["track"]
        laps = test["race_config"]["total_laps"]
        temp = test["race_config"]["track_temp"]
        config_key = f"{track} (L:{laps}, T:{temp}°)"
        config_errors[config_key] += 1
        config_totals[config_key] += 1
    
    print("Race configs with failures:")
    sorted_configs = sorted(config_totals.items(), key=lambda x: -x[1])
    for config, count in sorted_configs[:10]:
        print(f"  {config}: {count} failures")
    
    # Analyze tire strategy patterns in failures
    print("\n[FAILING CASES - TIRE STRATEGY PATTERNS]")
    strategy_patterns = defaultdict(int)
    
    for name, test, exp, pred in failing:
        expected_order = exp["finishing_positions"]
        
        # Find most common pit pattern among top 5 finishers
        top5_pit_counts = []
        for i in range(5):
            driver = expected_order[i]
            for pos in range(1, 21):
                if test["strategies"][f"pos{pos}"]["driver_id"] == driver:
                    pit_count = len(test["strategies"][f"pos{pos}"].get("pit_stops", []))
                    top5_pit_counts.append(pit_count)
                    break
        
        pattern = tuple(sorted(top5_pit_counts))
        strategy_patterns[pattern] += 1
    
    print("Top 5 finisher pit patterns (# stops each):")
    for pattern, count in sorted(strategy_patterns.items(), key=lambda x: -x[1])[:5]:
        print(f"  {pattern}: {count} races")
    
    # Check if there's a specific failing test to inspect
    print("\n[SAMPLE FAILING CASES TO INSPECT]")
    for i, (name, test, exp, pred) in enumerate(failing[:3]):
        print(f"\n{name}:")
        print(f"  Expected: {exp['finishing_positions'][:5]} ...top 5")
        print(f"  Predicted: {pred['finishing_positions'][:5]} ...top 5")
        print(f"  Race: {test['race_config']['track']} (L:{test['race_config']['total_laps']}, T:{test['race_config']['track_temp']}°)")
        
        # Find first position disagreement
        for pos in range(20):
            exp_driver = exp['finishing_positions'][pos]
            pred_driver = pred['finishing_positions'][pos]
            if exp_driver != pred_driver:
                pred_pos = pred['finishing_positions'].index(exp_driver)
                print(f"  First diff at position {pos+1}: expected {exp_driver} (pred has at {pred_pos+1})")
                break

if __name__ == "__main__":
    main()
