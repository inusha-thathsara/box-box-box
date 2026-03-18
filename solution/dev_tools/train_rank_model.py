#!/usr/bin/env python3
"""Train a regime-aware pairwise ranking model.

This script learns weights ``w`` such that for driver pair (i, j):
    (x_i - x_j) dot w > 0  =>  i should finish ahead of j.

The output is ``solution/rank_model.json`` used by ``race_simulator.py`` as an
optional second-stage correction on top of the physics simulator.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

from race_simulator import FEATURE_NAMES, build_feature_vector


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _regime_key(total_laps: int, track_temp: float, average_pit_count: float) -> str:
    if track_temp < 25.0:
        temp_bucket = "temp_low"
    elif track_temp <= 35.0:
        temp_bucket = "temp_mid"
    else:
        temp_bucket = "temp_high"

    if total_laps < 48:
        lap_bucket = "laps_short"
    elif total_laps < 60:
        lap_bucket = "laps_medium"
    else:
        lap_bucket = "laps_long"

    pit_bucket = "pit_low" if average_pit_count < 1.5 else "pit_high"
    return f"{temp_bucket}|{lap_bucket}|{pit_bucket}"


def _iter_historical_races(file_paths: Iterable[str]) -> Iterable[dict]:
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as file_obj:
            races = json.load(file_obj)
        for race in races:
            yield race


def _sample_pairs(
    indices: List[int],
    limit: int,
    rng: random.Random,
    max_rank_distance: int,
) -> List[Tuple[int, int]]:
    all_pairs = list(combinations(indices, 2))
    if max_rank_distance > 0:
        all_pairs = [pair for pair in all_pairs if abs(pair[1] - pair[0]) <= max_rank_distance]

    if not all_pairs:
        return []

    adjacent_pairs = [pair for pair in all_pairs if abs(pair[1] - pair[0]) == 1]
    non_adjacent_pairs = [pair for pair in all_pairs if abs(pair[1] - pair[0]) != 1]

    # Always keep adjacent pairs when possible; they are most important for exact ordering.
    if limit <= 0 or limit >= len(all_pairs):
        return all_pairs
    if len(adjacent_pairs) >= limit:
        return rng.sample(adjacent_pairs, limit)

    remaining = limit - len(adjacent_pairs)
    sampled = list(adjacent_pairs)
    if remaining > 0 and non_adjacent_pairs:
        if remaining >= len(non_adjacent_pairs):
            sampled.extend(non_adjacent_pairs)
        else:
            sampled.extend(rng.sample(non_adjacent_pairs, remaining))
    return sampled


def _update(
    weights: List[float],
    diff: List[float],
    lr: float,
    l2: float,
    clip_norm: float,
    sample_weight: float,
) -> None:
    margin = _dot(weights, diff)
    pred = _sigmoid(margin)
    error = pred - 1.0
    grads = [sample_weight * error * value + l2 * weights[idx] for idx, value in enumerate(diff)]

    if clip_norm > 0.0:
        grad_norm = _l2_norm(grads)
        if grad_norm > clip_norm:
            scale = clip_norm / (grad_norm + 1e-12)
            grads = [g * scale for g in grads]

    for idx, grad in enumerate(grads):
        weights[idx] -= lr * grad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train regime-aware pairwise rank model")
    parser.add_argument(
        "--data-glob",
        default="data/historical_races/races_*.json",
        help="Glob for historical race files",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--l2", type=float, default=1e-5, help="L2 regularization")
    parser.add_argument(
        "--pairs-per-race",
        type=int,
        default=120,
        help="Max sampled driver pairs per race (0 or large means all)",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument(
        "--min-regime-races",
        type=int,
        default=300,
        help="Minimum races before exporting a dedicated regime model",
    )
    parser.add_argument(
        "--alpha-seconds",
        type=float,
        default=0.22,
        help="Inference blend scale (seconds) for rank correction",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Inference top-k swap window")
    parser.add_argument("--max-swaps", type=int, default=8, help="Inference swap budget")
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=0.25,
        help="Only swap adjacent drivers when adjusted-time gap is within this threshold",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.05,
        help="Minimum normalized rank-score margin needed for a swap",
    )
    parser.add_argument(
        "--output",
        default="solution/rank_model.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=3.0,
        help="Gradient clipping L2 norm (<=0 disables clipping)",
    )
    parser.add_argument(
        "--normalize-diff",
        action="store_true",
        help="Normalize each pairwise feature difference to unit L2 norm",
    )
    parser.add_argument(
        "--distance-weight-power",
        type=float,
        default=0.0,
        help="Weight pairs by 1/(rank_distance^power); 0 disables distance weighting",
    )
    parser.add_argument(
        "--adjacent-multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier for adjacent (distance=1) pairs",
    )
    parser.add_argument(
        "--max-rank-distance",
        type=int,
        default=0,
        help="Keep only pairs with finish-rank distance <= this value (0 disables filter)",
    )
    parser.add_argument(
        "--topk-train-focus",
        type=int,
        default=0,
        help="Apply extra weight when either pair member is within top-k finish positions (0 disables)",
    )
    parser.add_argument(
        "--topk-train-multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier for top-k focused pair updates",
    )
    return parser.parse_args()


def _resolve_data_glob(pattern: str) -> List[str]:
    """Resolve historical data glob from common working directories.

    Supports running from either repository root or the solution/ folder.
    """

    candidates = [pattern]
    if not os.path.isabs(pattern):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(script_dir, pattern))
        candidates.append(os.path.join(script_dir, "..", pattern))

        # Convenience fallback when command uses data/... while cwd is solution/
        if pattern.startswith("data/") or pattern.startswith("data\\"):
            candidates.append(os.path.join("..", pattern))

    seen = set()
    resolved: List[str] = []
    for candidate in candidates:
        for path in sorted(glob.glob(candidate)):
            norm = os.path.normcase(os.path.abspath(os.path.normpath(path)))
            if norm in seen:
                continue
            seen.add(norm)
            resolved.append(path)
    return resolved


def main() -> None:
    args = parse_args()

    file_paths = _resolve_data_glob(args.data_glob)
    if not file_paths:
        raise SystemExit(f"No files matched: {args.data_glob}")

    feature_count = len(FEATURE_NAMES)
    rng = random.Random(args.seed)

    global_weights = [0.0] * feature_count
    regime_weights: Dict[str, List[float]] = {}
    regime_race_counts: Dict[str, int] = {}

    races = list(_iter_historical_races(file_paths))
    print(f"Loaded races: {len(races)} from {len(file_paths)} files")

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(races)
        total_pairs = 0

        for race in races:
            race_config = race["race_config"]
            strategies = race["strategies"]
            finish_order = race["finishing_positions"]

            average_pit_count = (
                sum(len(strategies[f"pos{i}"].get("pit_stops", [])) for i in range(1, 21)) / 20.0
            )
            regime = _regime_key(
                total_laps=int(race_config["total_laps"]),
                track_temp=float(race_config["track_temp"]),
                average_pit_count=average_pit_count,
            )
            if regime not in regime_weights:
                regime_weights[regime] = [0.0] * feature_count
                regime_race_counts[regime] = 0
            regime_race_counts[regime] += 1

            features_by_driver: Dict[str, List[float]] = {}
            for position in sorted(strategies.keys(), key=lambda key: int(key[3:])):
                strategy = strategies[position]
                driver_id = str(strategy["driver_id"])
                features_by_driver[driver_id] = build_feature_vector(race_config, strategy)

            indices = list(range(len(finish_order)))
            sampled_pairs = _sample_pairs(
                indices=indices,
                limit=args.pairs_per_race,
                rng=rng,
                max_rank_distance=args.max_rank_distance,
            )
            total_pairs += len(sampled_pairs)

            reg_w = regime_weights[regime]
            for left_idx, right_idx in sampled_pairs:
                better_driver = finish_order[left_idx]
                worse_driver = finish_order[right_idx]
                x_better = features_by_driver[better_driver]
                x_worse = features_by_driver[worse_driver]
                diff = [a - b for a, b in zip(x_better, x_worse)]

                distance = abs(right_idx - left_idx)
                sample_weight = 1.0
                if args.distance_weight_power > 0.0 and distance > 0:
                    sample_weight = 1.0 / (float(distance) ** args.distance_weight_power)
                if distance == 1 and args.adjacent_multiplier > 0.0:
                    sample_weight *= args.adjacent_multiplier
                if (
                    args.topk_train_focus > 0
                    and args.topk_train_multiplier > 0.0
                    and (left_idx < args.topk_train_focus or right_idx < args.topk_train_focus)
                ):
                    sample_weight *= args.topk_train_multiplier

                if args.normalize_diff:
                    norm = _l2_norm(diff)
                    if norm > 1e-12:
                        diff = [value / norm for value in diff]
                _update(reg_w, diff, args.lr, args.l2, args.clip_norm, sample_weight)
                _update(global_weights, diff, args.lr, args.l2, args.clip_norm, sample_weight)

        print(f"Epoch {epoch}/{args.epochs}: sampled_pairs={total_pairs}")

    export_regimes: Dict[str, Dict[str, object]] = {}
    regime_inference: Dict[str, Dict[str, float | int]] = {}
    for regime, weights in regime_weights.items():
        race_count = regime_race_counts.get(regime, 0)
        if race_count < args.min_regime_races:
            continue
        export_regimes[regime] = {
            "weights": weights,
            "race_count": race_count,
        }

        alpha_scale = 1.0
        if "laps_short" in regime:
            alpha_scale = 1.15
        elif "laps_long" in regime:
            alpha_scale = 0.90
        if "pit_high" in regime:
            alpha_scale *= 0.95

        regime_inference[regime] = {
            "alpha_seconds": args.alpha_seconds * alpha_scale,
            "top_k": args.top_k,
            "max_swaps": args.max_swaps,
            "max_gap_seconds": args.max_gap_seconds,
            "min_margin": args.min_margin,
        }

    payload = {
        "feature_names": list(FEATURE_NAMES),
        "global_weights": global_weights,
        "regimes": export_regimes,
        "inference": {
            "alpha_seconds": args.alpha_seconds,
            "top_k": args.top_k,
            "max_swaps": args.max_swaps,
            "max_gap_seconds": args.max_gap_seconds,
            "min_margin": args.min_margin,
        },
        "regime_inference": regime_inference,
        "training_info": {
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "clip_norm": args.clip_norm,
            "normalize_diff": bool(args.normalize_diff),
            "distance_weight_power": args.distance_weight_power,
            "adjacent_multiplier": args.adjacent_multiplier,
            "max_rank_distance": args.max_rank_distance,
            "topk_train_focus": args.topk_train_focus,
            "topk_train_multiplier": args.topk_train_multiplier,
            "pairs_per_race": args.pairs_per_race,
            "seed": args.seed,
            "total_races": len(races),
            "historical_files": len(file_paths),
            "min_regime_races": args.min_regime_races,
        },
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
    print(f"Wrote rank model: {args.output}")
    print(f"Exported regime models: {len(export_regimes)}")


if __name__ == "__main__":
    main()
