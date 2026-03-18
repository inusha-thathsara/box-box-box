#!/usr/bin/env python3
"""Train a simple linear model to predict driver finishing order.

This approach:
1. Directly learns to rank drivers based on features
2. Uses stochastic gradient descent on pairwise ranking loss
3. Saves learned weights to linear_model.json
"""

import json
import sys
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Must match race_simulator.py FEATURE_NAMES (44 dims)
FEATURE_NAMES = (
    "base_lap_time", "pit_lane_time", "track_temp", "total_laps",
    "start_soft", "start_medium", "start_hard",
    "pit_count", "laps_soft", "laps_medium", "laps_hard",
    "age_sum_soft", "age_sum_medium", "age_sum_hard",
    "age_sq_sum_soft", "age_sq_sum_medium", "age_sq_sum_hard",
    "last_soft", "last_medium", "last_hard",
    "stint_count", "avg_pit_lap", "first_pit_early",
    "driver_D001", "driver_D002", "driver_D003", "driver_D004", "driver_D005",
    "driver_D006", "driver_D007", "driver_D008", "driver_D009", "driver_D010",
    "driver_D011", "driver_D012", "driver_D013", "driver_D014", "driver_D015",
    "driver_D016", "driver_D017", "driver_D018", "driver_D019", "driver_D020",
)


def load_historical_races(glob_pattern):
    """Load all historical races from JSON files."""
    from pathlib import Path
    
    races = []
    
    # Try both relative paths (from root and from solution folder)
    for base_path in [Path("."), Path("..")]:
        files = sorted(base_path.glob(glob_pattern))
        if files:
            for file_path in files:
                try:
                    with open(file_path) as f:
                        file_races = json.load(f)
                        races.extend(file_races)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}", file=sys.stderr)
            break
    
    return races


def build_feature_vector(race_config, strategy):
    """Build 44-dim feature vector matching race_simulator.py."""
    total_laps = int(race_config["total_laps"])
    start_tire = str(strategy["starting_tire"]).upper()
    driver_id = str(strategy.get("driver_id", ""))
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))
    
    laps_by_compound = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    age_sum = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    age_sq_sum = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    
    pit_schedule = {int(s["lap"]): s["to_tire"].upper() for s in pit_stops}
    current = start_tire
    age = 0
    
    for lap in range(1, total_laps + 1):
        age += 1
        laps_by_compound[current] += 1.0
        age_sum[current] += float(age)
        age_sq_sum[current] += float(age * age)
        if lap in pit_schedule:
            current = pit_schedule[lap]
            age = 0
    
    pit_laps = [int(s["lap"]) for s in pit_stops]
    avg_pit_lap = sum(pit_laps) / len(pit_laps) if pit_laps else 0.0
    first_pit_early = 1.0 if pit_laps and pit_laps[0] <= 10 else 0.0
    
    base_features = [
        float(race_config["base_lap_time"]),
        float(race_config["pit_lane_time"]),
        float(race_config["track_temp"]),
        float(total_laps),
        1.0 if start_tire == "SOFT" else 0.0,
        1.0 if start_tire == "MEDIUM" else 0.0,
        1.0 if start_tire == "HARD" else 0.0,
        float(len(pit_stops)),
        laps_by_compound["SOFT"],
        laps_by_compound["MEDIUM"],
        laps_by_compound["HARD"],
        age_sum["SOFT"],
        age_sum["MEDIUM"],
        age_sum["HARD"],
        age_sq_sum["SOFT"],
        age_sq_sum["MEDIUM"],
        age_sq_sum["HARD"],
        1.0 if current == "SOFT" else 0.0,
        1.0 if current == "MEDIUM" else 0.0,
        1.0 if current == "HARD" else 0.0,
        float(len(pit_stops)) + 1.0,
        avg_pit_lap,
        first_pit_early,
    ]
    
    driver_one_hot = [1.0 if driver_id == f"D{i:03d}" else 0.0 for i in range(1, 21)]
    return base_features + driver_one_hot


def sigmoid(x):
    """Sigmoid activation."""
    return 1.0 / (1.0 + math.exp(-max(-100, min(100, x))))


def train_linear_model(races, epochs=5, lr=0.01, l2=0.0001, clip_norm=2.0):
    """Train linear weights via pairwise SGD."""
    
    feature_dim = len(FEATURE_NAMES)
    weights = [0.0] * feature_dim
    bias = 0.0
    
    print(f"Training linear model: {epochs} epochs, lr={lr}, l2={l2}", file=sys.stderr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        pair_count = 0
        
        shuffled_races = races[:]
        random.shuffle(shuffled_races)
        
        for race in shuffled_races:
            race_config = race["race_config"]
            strategies = race["strategies"]
            finish_order = race.get("finishing_positions", [])
            
            if not finish_order or len(finish_order) < 2:
                continue
            
            # Build features for each driver
            driver_features = {}
            for pos in range(1, 21):
                strategy = strategies[f"pos{pos}"]
                driver_id = strategy["driver_id"]
                features = build_feature_vector(race_config, strategy)
                driver_features[driver_id] = features
            
            # Sample random pairs and apply pairwise loss
            for i in range(min(50, len(finish_order) - 1)):
                idx1 = random.randint(0, len(finish_order) - 2)
                idx2 = random.randint(idx1 + 1, len(finish_order) - 1)
                
                driver1 = finish_order[idx1]
                driver2 = finish_order[idx2]
                
                if driver1 not in driver_features or driver2 not in driver_features:
                    continue
                
                feat1 = driver_features[driver1]
                feat2 = driver_features[driver2]
                
                # Score difference (higher = better)
                score1 = bias + sum(w * f for w, f in zip(weights, feat1))
                score2 = bias + sum(w * f for w, f in zip(weights, feat2))
                margin = score1 - score2
                
                # Pairwise sigmoid loss: driver1 should rank higher
                prob = sigmoid(margin)
                loss = -math.log(max(1e-9, prob))
                total_loss += loss
                pair_count += 1
                
                # Gradient step
                grad_factor = (prob - 1.0) * lr
                
                for j in range(feature_dim):
                    grad = grad_factor * (feat1[j] - feat2[j]) + l2 * weights[j]
                    weights[j] -= grad
                
                bias -= grad_factor * lr
        
        # Clip gradients
        norm = math.sqrt(sum(w * w for w in weights))
        if norm > clip_norm:
            scale = clip_norm / norm
            weights = [w * scale for w in weights]
        
        avg_loss = total_loss / max(1, pair_count)
        print(f"Epoch {epoch + 1}/{epochs}: {pair_count} pairs, loss={avg_loss:.4f}", file=sys.stderr)
    
    return weights, bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-glob", default="../data/historical_races/races_*.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--clip-norm", type=float, default=2.0)
    parser.add_argument("--output", default="linear_model.json")
    
    args = parser.parse_args()
    
    # Load races
    races = load_historical_races(args.data_glob)
    print(f"Loaded {len(races)} races", file=sys.stderr)
    
    if not races:
        print("ERROR: No races loaded", file=sys.stderr)
        sys.exit(1)
    
    # Train
    weights, bias = train_linear_model(
        races,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        clip_norm=args.clip_norm,
    )
    
    # Save
    output = {
        "feature_names": list(FEATURE_NAMES),
        "weights": weights,
        "bias": bias,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved model to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
