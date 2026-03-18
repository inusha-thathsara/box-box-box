#!/usr/bin/env python3
"""Neural network trainer for F1 race ranking.

Trains a small 2-layer NN to predict driver scores from 44-dim features.
Loss: Pairwise ranking (score(better_driver) > score(worse_driver)).

Run: python train_nn_model.py --epochs 10 --lr 0.001 --output nn_model.pth
"""

import json
import sys
import math
import random
import argparse
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch", file=sys.stderr)
    sys.exit(1)


class RankingNetwork(nn.Module):
    """Simple 2-layer network for ranking drivers."""
    
    def __init__(self, input_dim=43):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_historical_races(glob_pattern):
    """Load all historical races from JSON files."""
    races = []
    
    # Try both relative paths
    for base_path in [Path("."), Path("..")]:
        files = sorted(base_path.glob(glob_pattern))
        if files:
            print(f"Found {len(files)} files in {base_path}")
            for file_path in files:
                try:
                    with open(file_path) as f:
                        file_races = json.load(f)
                        races.extend(file_races)
                        print(f"  Loaded {len(file_races)} races from {file_path.name}")
                except Exception as e:
                    print(f"  Error: {e}", file=sys.stderr)
            break
    
    return races


def build_feature_vector(race_config, strategy):
    """Build 43-dim feature vector (matches race_simulator.py exactly)."""
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
        float(race_config["base_lap_time"]),  # 1
        float(race_config["pit_lane_time"]),  # 2
        float(race_config["track_temp"]),  # 3
        float(total_laps),  # 4
        1.0 if start_tire == "SOFT" else 0.0,  # 5
        1.0 if start_tire == "MEDIUM" else 0.0,  # 6
        1.0 if start_tire == "HARD" else 0.0,  # 7
        float(len(pit_stops)),  # 8
        laps_by_compound["SOFT"],  # 9
        laps_by_compound["MEDIUM"],  # 10
        laps_by_compound["HARD"],  # 11
        age_sum["SOFT"],  # 12
        age_sum["MEDIUM"],  # 13
        age_sum["HARD"],  # 14
        age_sq_sum["SOFT"],  # 15
        age_sq_sum["MEDIUM"],  # 16
        age_sq_sum["HARD"],  # 17
        1.0 if current == "SOFT" else 0.0,  # 18
        1.0 if current == "MEDIUM" else 0.0,  # 19
        1.0 if current == "HARD" else 0.0,  # 20
        float(len(pit_stops)) + 1.0,  # 21 (stint_count)
        avg_pit_lap,  # 22
        first_pit_early,  # 23
    ]
    
    driver_one_hot = [1.0 if driver_id == f"D{i:03d}" else 0.0 for i in range(1, 21)]  # 24-43 (20 drivers)
    features = base_features + driver_one_hot
    assert len(features) == 43, f"Expected 43 features, got {len(features)}"
    return features


def train_nn(races, epochs=10, lr=0.001, batch_pairs=500, device="cpu"):
    """Train neural network on historical races."""
    
    print(f"Training on {len(races)} races, device={device}")
    print(f"Epochs: {epochs}, LR: {lr}, Batch pairs: {batch_pairs}")
    print("=" * 80)
    
    model = RankingNetwork(input_dim=43).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        pair_count = 0
        
        # Shuffle races
        shuffled_races = races[:]
        random.shuffle(shuffled_races)
        
        for race in shuffled_races:
            race_config = race["race_config"]
            strategies = race["strategies"]
            finish_order = race.get("finishing_positions", [])
            
            if not finish_order or len(finish_order) < 2:
                continue
            
            # Build features for all drivers
            driver_features = {}
            for pos in range(1, 21):
                strategy = strategies[f"pos{pos}"]
                driver_id = strategy["driver_id"]
                features = build_feature_vector(race_config, strategy)
                driver_features[driver_id] = torch.tensor(features, dtype=torch.float32, device=device)
            
            # Generate pairwise comparisons
            for _ in range(min(batch_pairs, len(finish_order) - 1)):
                idx1 = random.randint(0, len(finish_order) - 2)
                idx2 = random.randint(idx1 + 1, len(finish_order) - 1)
                
                driver1 = finish_order[idx1]  # Should rank higher
                driver2 = finish_order[idx2]  # Should rank lower
                
                if driver1 not in driver_features or driver2 not in driver_features:
                    continue
                
                feat1 = driver_features[driver1]
                feat2 = driver_features[driver2]
                
                # Forward pass
                score1 = model(feat1.unsqueeze(0))
                score2 = model(feat2.unsqueeze(0))
                
                # Margin loss: want score1 - score2 > 0 (margin = 0.1)
                margin = 0.1
                loss = torch.nn.functional.relu(margin - (score1 - score2))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pair_count += 1
        
        avg_loss = total_loss / max(1, pair_count)
        print(f"Epoch {epoch + 1}/{epochs}: {pair_count} pairs, loss={avg_loss:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-glob", default="../data/historical_races/races_*.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-pairs", type=int, default=500)
    parser.add_argument("--output", default="nn_model.pth")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    
    args = parser.parse_args()
    
    # Load races
    print("=" * 80)
    print("NEURAL NETWORK RACE RANKING TRAINER")
    print("=" * 80)
    
    races = load_historical_races(args.data_glob)
    print(f"\nTotal loaded: {len(races)} races")
    
    if not races:
        print("ERROR: No races loaded", file=sys.stderr)
        sys.exit(1)
    
    # Train
    print()
    device = torch.device(args.device)
    model = train_nn(
        races,
        epochs=args.epochs,
        lr=args.lr,
        batch_pairs=args.batch_pairs,
        device=device
    )
    
    # Save
    print()
    print("=" * 80)
    
    # Save as CPU model so it can run anywhere
    model_cpu = RankingNetwork(input_dim=43).to("cpu")
    model_cpu.load_state_dict(model.cpu().state_dict())
    
    torch.save(model_cpu.state_dict(), args.output)
    print(f"Saved model to {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
