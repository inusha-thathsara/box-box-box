#!/usr/bin/env python3
"""Train a data-driven matching model from historical races.

The model stores backoff lookup tables keyed by race+strategy signatures.
At inference time, race_simulator.py can load match_model.json and predict
finish order by expected finishing position from nearest matching patterns.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _build_match_keys(race_config: dict, strategy: dict) -> Dict[str, str]:
    driver_id = str(strategy.get("driver_id", ""))
    total_laps = int(race_config["total_laps"])

    race_exact = _race_signature(race_config, coarse=False)
    race_coarse = _race_signature(race_config, coarse=True)
    strat_exact = _strategy_signature(strategy, total_laps=total_laps, coarse=False)
    strat_coarse = _strategy_signature(strategy, total_laps=total_laps, coarse=True)

    return {
        "exact": f"{driver_id}|{race_exact}|{strat_exact}",
        "driver_strategy": f"{driver_id}|{strat_coarse}",
        "coarse": f"{race_coarse}|{strat_coarse}",
        "driver_race": f"{driver_id}|{race_coarse}",
        "driver": driver_id,
    }


def _update_table(table: Dict[str, List[float]], key: str, finish_pos: int) -> None:
    if key not in table:
        table[key] = [0.0, 0.0]
    table[key][0] += float(finish_pos)
    table[key][1] += 1.0


def load_historical_races(glob_pattern: str) -> List[dict]:
    races: List[dict] = []
    files = sorted(Path(".").glob(glob_pattern))
    if not files:
        files = sorted(Path("..").glob(glob_pattern))

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as file_obj:
            loaded = json.load(file_obj)
            races.extend(loaded)
    return races


def train_match_model(races: List[dict]) -> Dict[str, object]:
    tables: Dict[str, Dict[str, List[float]]] = {
        "exact": {},
        "driver_strategy": {},
        "coarse": {},
        "driver_race": {},
        "driver": {},
    }

    total_positions = 0.0
    total_samples = 0.0

    for race in races:
        race_config = race["race_config"]
        strategies = race["strategies"]
        finish_order = race.get("finishing_positions", [])
        if not finish_order:
            continue

        finish_pos_by_driver = {
            str(driver_id): int(idx + 1)
            for idx, driver_id in enumerate(finish_order)
        }

        for pos_key in sorted(strategies.keys(), key=lambda key: int(key[3:])):
            strategy = strategies[pos_key]
            driver_id = str(strategy["driver_id"])
            finish_pos = finish_pos_by_driver.get(driver_id)
            if finish_pos is None:
                continue

            keys = _build_match_keys(race_config, strategy)
            for table_name, table_key in keys.items():
                _update_table(tables[table_name], table_key, finish_pos)

            total_positions += float(finish_pos)
            total_samples += 1.0

    for table in tables.values():
        for key, value in table.items():
            sum_pos = float(value[0])
            count = float(value[1])
            mean_pos = sum_pos / count if count > 0.0 else 10.5
            table[key] = [mean_pos, int(count)]

    global_mean = (total_positions / total_samples) if total_samples > 0.0 else 10.5

    return {
        "model_type": "data_driven_matching_v1",
        "global_mean_pos": global_mean,
        "tables": tables,
        "stats": {
            "race_count": len(races),
            "sample_count": int(total_samples),
            "table_sizes": {name: len(table) for name, table in tables.items()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train data-driven matching model")
    parser.add_argument(
        "--data-glob",
        default="../data/historical_races/races_*.json",
        help="Glob for historical race json files",
    )
    parser.add_argument(
        "--output",
        default="match_model.json",
        help="Output JSON model path",
    )
    args = parser.parse_args()

    print("Loading historical races...")
    races = load_historical_races(args.data_glob)
    if not races:
        raise SystemExit("No historical races found")
    print(f"Loaded races: {len(races)}")

    print("Training data-driven matching model...")
    model = train_match_model(races)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(model, file_obj, separators=(",", ":"))

    stats = model.get("stats", {})
    print(f"Saved model to {output_path}")
    print(f"Samples: {stats.get('sample_count', 0)}")
    table_sizes = stats.get("table_sizes", {})
    if isinstance(table_sizes, dict):
        print(
            "Table sizes: "
            + ", ".join(
                f"{name}={size}" for name, size in sorted(table_sizes.items())
            )
        )


if __name__ == "__main__":
    main()
