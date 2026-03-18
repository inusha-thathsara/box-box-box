#!/usr/bin/env python3
"""Train pairwise data-driven matching model from historical races."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _bucket(value: float, step: float) -> int:
    return int(round(value / step))


def _pit_phase(lap: int, total_laps: int) -> str:
    if total_laps <= 0:
        return "M"
    ratio = float(lap) / float(total_laps)
    if ratio <= 0.33:
        return "E"
    if ratio <= 0.66:
        return "M"
    return "L"


def _race_signature(race_config: dict, coarse: bool) -> str:
    base_step = 0.5 if not coarse else 1.0
    pit_step = 0.25 if not coarse else 0.5
    temp_step = 2.0 if not coarse else 4.0
    laps_step = 2.0 if not coarse else 4.0

    return "|".join(
        (
            f"b{_bucket(float(race_config['base_lap_time']), base_step)}",
            f"p{_bucket(float(race_config['pit_lane_time']), pit_step)}",
            f"t{_bucket(float(race_config['track_temp']), temp_step)}",
            f"l{_bucket(float(race_config['total_laps']), laps_step)}",
        )
    )


def _strategy_signature(strategy: dict, total_laps: int, coarse: bool) -> str:
    start_tire = str(strategy.get("starting_tire", "")).upper()
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda stop: int(stop["lap"]))
    pit_count = len(pit_stops)

    to_tires = ",".join(str(stop["to_tire"]).upper() for stop in pit_stops) or "NONE"
    if pit_stops:
        avg_pit_lap = sum(int(stop["lap"]) for stop in pit_stops) / float(pit_count)
        first_lap = int(pit_stops[0]["lap"])
    else:
        avg_pit_lap = 0.0
        first_lap = 0

    avg_bucket = _bucket(avg_pit_lap, 2.0 if not coarse else 4.0)
    first_phase = _pit_phase(first_lap, total_laps) if pit_stops else "N"

    if coarse:
        laps_part = "".join(_pit_phase(int(stop["lap"]), total_laps) for stop in pit_stops) or "N"
    else:
        laps_part = ",".join(str(int(stop["lap"])) for stop in pit_stops) or "NONE"

    final_tire = str(pit_stops[-1]["to_tire"]).upper() if pit_stops else start_tire
    return "|".join(
        (
            start_tire,
            f"pc{pit_count}",
            f"fp{first_phase}",
            f"ap{avg_bucket}",
            f"pt{laps_part}",
            f"tt{to_tires}",
            f"lt{final_tire}",
        )
    )


def _load_historical_races(glob_pattern: str) -> List[dict]:
    races: List[dict] = []
    files = sorted(Path(".").glob(glob_pattern))
    if not files:
        files = sorted(Path("..").glob(glob_pattern))

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as file_obj:
            races.extend(json.load(file_obj))
    return races


def _update_pair_table(table: Dict[str, List[float]], key: str, left_win: float) -> None:
    if key not in table:
        table[key] = [0.0, 0.0]
    table[key][0] += left_win
    table[key][1] += 1.0


def _finalize_pair_table(table: Dict[str, List[float]]) -> Dict[str, List[float]]:
    finalized: Dict[str, List[float]] = {}
    for key, (sum_left_win, count) in table.items():
        if count <= 0:
            continue
        finalized[key] = [float(sum_left_win) / float(count), int(count)]
    return finalized


def train_pair_model(races: List[dict]) -> Dict[str, object]:
    pair_driver_race_raw: Dict[str, List[float]] = {}
    pair_strategy_race_raw: Dict[str, List[float]] = {}
    pair_driver_strategy_raw: Dict[str, List[float]] = {}

    for race in races:
        race_config = race["race_config"]
        strategies = race["strategies"]
        finish_order = race.get("finishing_positions", [])
        if len(finish_order) < 2:
            continue

        race_coarse = _race_signature(race_config, coarse=True)
        total_laps = int(race_config["total_laps"])

        strategy_by_driver: Dict[str, dict] = {}
        strategy_sig_by_driver: Dict[str, str] = {}
        for pos_key in sorted(strategies.keys(), key=lambda key: int(key[3:])):
            strategy = strategies[pos_key]
            driver_id = str(strategy["driver_id"])
            strategy_by_driver[driver_id] = strategy
            strategy_sig_by_driver[driver_id] = _strategy_signature(strategy, total_laps=total_laps, coarse=True)

        for i in range(len(finish_order) - 1):
            winner = str(finish_order[i])
            winner_sig = strategy_sig_by_driver.get(winner)
            if winner_sig is None:
                continue
            for j in range(i + 1, len(finish_order)):
                loser = str(finish_order[j])
                loser_sig = strategy_sig_by_driver.get(loser)
                if loser_sig is None:
                    continue

                # Driver pair by race bucket.
                left_driver, right_driver = sorted((winner, loser))
                driver_key = f"{race_coarse}|{left_driver}|{right_driver}"
                left_win = 1.0 if winner == left_driver else 0.0
                _update_pair_table(pair_driver_race_raw, driver_key, left_win)

                # Strategy pair by race bucket.
                left_sig, right_sig = sorted((winner_sig, loser_sig))
                strategy_key = f"{race_coarse}|{left_sig}|{right_sig}"
                left_sig_win = 1.0 if winner_sig == left_sig else 0.0
                _update_pair_table(pair_strategy_race_raw, strategy_key, left_sig_win)

                # Driver+strategy pair global.
                winner_pair = f"{winner}|{winner_sig}"
                loser_pair = f"{loser}|{loser_sig}"
                left_pair, right_pair = sorted((winner_pair, loser_pair))
                pair_key = f"{left_pair}|{right_pair}"
                left_pair_win = 1.0 if winner_pair == left_pair else 0.0
                _update_pair_table(pair_driver_strategy_raw, pair_key, left_pair_win)

    pair_driver_race = _finalize_pair_table(pair_driver_race_raw)
    pair_strategy_race = _finalize_pair_table(pair_strategy_race_raw)
    pair_driver_strategy = _finalize_pair_table(pair_driver_strategy_raw)

    return {
        "model_type": "pair_data_driven_matching_v1",
        "top_k": 10,
        "max_swaps": 4,
        "min_bias": 1.4,
        "tables": {
            "pair_driver_race": pair_driver_race,
            "pair_strategy_race": pair_strategy_race,
            "pair_driver_strategy": pair_driver_strategy,
        },
        "stats": {
            "race_count": len(races),
            "pair_driver_race": len(pair_driver_race),
            "pair_strategy_race": len(pair_strategy_race),
            "pair_driver_strategy": len(pair_driver_strategy),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pairwise data-driven matching model")
    parser.add_argument(
        "--data-glob",
        default="../data/historical_races/races_*.json",
        help="Glob for historical race json files",
    )
    parser.add_argument(
        "--output",
        default="pair_match_model.json",
        help="Output JSON model path",
    )
    args = parser.parse_args()

    print("Loading historical races...")
    races = _load_historical_races(args.data_glob)
    if not races:
        raise SystemExit("No historical races found")
    print(f"Loaded races: {len(races)}")

    print("Training pairwise data-driven matching model...")
    model = train_pair_model(races)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(model, file_obj, separators=(",", ":"))

    stats = model.get("stats", {})
    print(f"Saved model to {output_path}")
    print(
        "Table sizes: "
        + ", ".join(
            f"{k}={v}" for k, v in stats.items() if k != "race_count"
        )
    )


if __name__ == "__main__":
    main()
