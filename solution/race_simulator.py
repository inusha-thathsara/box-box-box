#!/usr/bin/env python3
"""Box Box Box race simulator.

This implementation provides a deterministic, parameterized lap-time model.
Tune the parameters using ``analyze.py`` and optionally store them in
``solution/model_params.json`` for improved accuracy.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

COMPOUNDS: Tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
FEATURE_NAMES: Tuple[str, ...] = (
    "base_lap_time",
    "pit_lane_time",
    "track_temp",
    "total_laps",
    "start_soft",
    "start_medium",
    "start_hard",
    "pit_count",
    "laps_soft",
    "laps_medium",
    "laps_hard",
    "age_sum_soft",
    "age_sum_medium",
    "age_sum_hard",
    "age_sq_sum_soft",
    "age_sq_sum_medium",
    "age_sq_sum_hard",
    "last_soft",
    "last_medium",
    "last_hard",
    "stint_count",
    "avg_pit_lap",
    "first_pit_early",
    "driver_D001",
    "driver_D002",
    "driver_D003",
    "driver_D004",
    "driver_D005",
    "driver_D006",
    "driver_D007",
    "driver_D008",
    "driver_D009",
    "driver_D010",
    "driver_D011",
    "driver_D012",
    "driver_D013",
    "driver_D014",
    "driver_D015",
    "driver_D016",
    "driver_D017",
    "driver_D018",
    "driver_D019",
    "driver_D020",
)


@dataclass(frozen=True)
class CompoundParams:
    """Tunable parameters for one tire compound."""

    base_offset: float
    grace_laps: int
    deg_linear: float
    deg_quadratic: float
    temp_base_coeff: float
    temp_deg_coeff: float
    late_start: int = 99
    late_deg_linear: float = 0.0


@dataclass(frozen=True)
class ModelParams:
    """Complete model definition used by the simulator."""

    temp_reference: float
    compounds: Dict[str, CompoundParams]
    fresh_tire_penalty: float = 0.0
    pre_pit_lap_offset: float = 0.0
    post_pit_lap_offset: float = 0.0
    short_race_pit_lane_scale: float = 1.0
    short_race_start_soft_bias: float = 0.0
    short_race_start_hard_bias: float = 0.0
    short_regime_rank_calibrator: Dict[str, Any] | None = None
    driver_offsets: Dict[str, float] | None = None
    driver_temp_coeffs: Dict[str, float] | None = None


@dataclass(frozen=True)
class RankCorrectionModel:
    """Optional learned ranking correction applied after lap-time simulation."""

    feature_names: Tuple[str, ...]
    global_weights: Tuple[float, ...]
    regime_weights: Dict[str, Tuple[float, ...]]
    regime_inference: Dict[str, Dict[str, float | int]] | None = None
    alpha_seconds: float = 0.20
    top_k: int = 10
    max_swaps: int = 8
    max_gap_seconds: float = 0.25
    min_margin: float = 0.05


DEFAULT_PARAMS = ModelParams(
    temp_reference=30.0,
    compounds={
        "SOFT": CompoundParams(
            base_offset=-0.52,
            grace_laps=2,
            deg_linear=0.044,
            deg_quadratic=0.0018,
            temp_base_coeff=-0.004,
            temp_deg_coeff=0.0013,
            late_start=16,
            late_deg_linear=0.010,
        ),
        "MEDIUM": CompoundParams(
            base_offset=0.0,
            grace_laps=3,
            deg_linear=0.029,
            deg_quadratic=0.0010,
            temp_base_coeff=-0.002,
            temp_deg_coeff=0.0008,
            late_start=22,
            late_deg_linear=0.006,
        ),
        "HARD": CompoundParams(
            base_offset=0.48,
            grace_laps=4,
            deg_linear=0.020,
            deg_quadratic=0.00065,
            temp_base_coeff=-0.001,
            temp_deg_coeff=0.00055,
            late_start=28,
            late_deg_linear=0.004,
        ),
    },
    fresh_tire_penalty=0.0,
    short_race_pit_lane_scale=1.0,
)


def _clamp_positive(value: float) -> float:
    return value if value > 0.0 else 0.0


def _load_params_from_file(path: str) -> ModelParams:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    compounds: Dict[str, CompoundParams] = {}
    for compound in COMPOUNDS:
        raw = payload["compounds"][compound]
        compounds[compound] = CompoundParams(
            base_offset=float(raw["base_offset"]),
            grace_laps=int(raw["grace_laps"]),
            deg_linear=float(raw["deg_linear"]),
            deg_quadratic=float(raw["deg_quadratic"]),
            temp_base_coeff=float(raw["temp_base_coeff"]),
            temp_deg_coeff=float(raw["temp_deg_coeff"]),
            late_start=int(raw.get("late_start", 99)),
            late_deg_linear=float(raw.get("late_deg_linear", 0.0)),
        )

    driver_offsets_payload = payload.get("driver_offsets", {})
    driver_temp_payload = payload.get("driver_temp_coeffs", {})

    return ModelParams(
        temp_reference=float(payload["temp_reference"]),
        compounds=compounds,
        fresh_tire_penalty=float(payload.get("fresh_tire_penalty", 0.0)),
        pre_pit_lap_offset=float(payload.get("pre_pit_lap_offset", 0.0)),
        post_pit_lap_offset=float(payload.get("post_pit_lap_offset", 0.0)),
        short_race_pit_lane_scale=float(payload.get("short_race_pit_lane_scale", 1.0)),
        short_race_start_soft_bias=float(payload.get("short_race_start_soft_bias", 0.0)),
        short_race_start_hard_bias=float(payload.get("short_race_start_hard_bias", 0.0)),
        short_regime_rank_calibrator=payload.get("short_regime_rank_calibrator"),
        driver_offsets={
            str(key): float(value)
            for key, value in driver_offsets_payload.items()
        }
        if isinstance(driver_offsets_payload, dict)
        else None,
        driver_temp_coeffs={
            str(key): float(value)
            for key, value in driver_temp_payload.items()
        }
        if isinstance(driver_temp_payload, dict)
        else None,
    )


def load_model_params() -> ModelParams:
    """Load params from disk if present, otherwise use defaults."""

    params_path = os.path.join(os.path.dirname(__file__), "model_params.json")
    if not os.path.exists(params_path):
        return DEFAULT_PARAMS
    try:
        return _load_params_from_file(params_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return DEFAULT_PARAMS


def _load_rank_model_from_file(path: str) -> RankCorrectionModel:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    raw_feature_names = payload.get("feature_names", list(FEATURE_NAMES))
    feature_names = tuple(str(name) for name in raw_feature_names)
    expected_len = len(feature_names)

    global_raw = payload.get("global_weights", [])
    if len(global_raw) != expected_len:
        raise ValueError("rank model global_weights length mismatch")
    global_weights = tuple(float(v) for v in global_raw)

    regime_weights: Dict[str, Tuple[float, ...]] = {}
    regimes_payload = payload.get("regimes", {})
    if isinstance(regimes_payload, dict):
        for regime_key, regime_info in regimes_payload.items():
            weights_raw = []
            if isinstance(regime_info, dict):
                weights_raw = regime_info.get("weights", [])
            elif isinstance(regime_info, list):
                weights_raw = regime_info
            if len(weights_raw) != expected_len:
                continue
            regime_weights[str(regime_key)] = tuple(float(v) for v in weights_raw)

    regime_inference_payload = payload.get("regime_inference", {})
    regime_inference: Dict[str, Dict[str, float | int]] = {}
    if isinstance(regime_inference_payload, dict):
        for regime_key, info in regime_inference_payload.items():
            if not isinstance(info, dict):
                continue
            regime_inference[str(regime_key)] = {
                "alpha_seconds": float(info.get("alpha_seconds", 0.20)),
                "top_k": int(info.get("top_k", 10)),
                "max_swaps": int(info.get("max_swaps", 8)),
                "max_gap_seconds": float(info.get("max_gap_seconds", 0.25)),
                "min_margin": float(info.get("min_margin", 0.05)),
            }

    inference = payload.get("inference", {})
    alpha_seconds = float(inference.get("alpha_seconds", 0.20))
    top_k = int(inference.get("top_k", 10))
    max_swaps = int(inference.get("max_swaps", 8))
    max_gap_seconds = float(inference.get("max_gap_seconds", 0.25))
    min_margin = float(inference.get("min_margin", 0.05))

    return RankCorrectionModel(
        feature_names=feature_names,
        global_weights=global_weights,
        regime_weights=regime_weights,
        regime_inference=regime_inference if regime_inference else None,
        alpha_seconds=alpha_seconds,
        top_k=top_k,
        max_swaps=max_swaps,
        max_gap_seconds=max_gap_seconds,
        min_margin=min_margin,
    )


def load_rank_model() -> RankCorrectionModel | None:
    """Load rank correction model if available. Returns None on any failure."""

    model_path = os.path.join(os.path.dirname(__file__), "rank_model.json")
    if not os.path.exists(model_path):
        return None
    try:
        return _load_rank_model_from_file(model_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _build_pit_schedule(pit_stops: Iterable[dict]) -> Dict[int, str]:
    schedule: Dict[int, str] = {}
    for stop in pit_stops:
        lap = int(stop["lap"])
        schedule[lap] = str(stop["to_tire"]).upper()
    return schedule


def _lap_time(
    base_lap_time: float,
    track_temp: float,
    tire: str,
    tire_age: int,
    params: ModelParams,
    lap_adjustment: float = 0.0,
) -> float:
    compound = params.compounds[tire]
    temp_delta = track_temp - params.temp_reference
    wear_age = max(0, tire_age - compound.grace_laps)
    late_wear_age = max(0, tire_age - compound.late_start)

    degradation = (
        compound.deg_linear * wear_age
        + compound.deg_quadratic * (wear_age * wear_age)
        + compound.temp_deg_coeff * temp_delta * wear_age
        + compound.late_deg_linear * late_wear_age
    )

    lap = base_lap_time + compound.base_offset + compound.temp_base_coeff * temp_delta
    if tire_age == 1:
        lap += params.fresh_tire_penalty
    return lap + _clamp_positive(degradation) + lap_adjustment


def build_feature_vector(race_config: dict, strategy: dict) -> List[float]:
    """Build a deterministic feature vector for one driver strategy."""

    total_laps = int(race_config["total_laps"])
    start_tire = str(strategy["starting_tire"]).upper()
    driver_id = str(strategy.get("driver_id", ""))
    pit_stops = sorted(strategy.get("pit_stops", []), key=lambda s: int(s["lap"]))

    laps_by_compound = {name: 0.0 for name in COMPOUNDS}
    age_sum_by_compound = {name: 0.0 for name in COMPOUNDS}
    age_sq_sum_by_compound = {name: 0.0 for name in COMPOUNDS}

    current = start_tire
    age = 0
    pit_schedule = _build_pit_schedule(pit_stops)
    for lap in range(1, total_laps + 1):
        age += 1
        laps_by_compound[current] += 1.0
        age_sum_by_compound[current] += float(age)
        age_sq_sum_by_compound[current] += float(age * age)
        if lap in pit_schedule:
            current = pit_schedule[lap]
            age = 0

    last_compound = current
    pit_count = float(len(pit_stops))
    stint_count = pit_count + 1.0
    
    # Pit timing features
    pit_laps = [int(stop["lap"]) for stop in pit_stops]
    avg_pit_lap = sum(pit_laps) / float(len(pit_laps)) if pit_laps else 0.0
    first_pit_early = 1.0 if pit_laps and pit_laps[0] <= 10 else 0.0

    def one_hot(compound: str, target: str) -> float:
        return 1.0 if compound == target else 0.0

    driver_one_hot = [1.0 if driver_id == f"D{i:03d}" else 0.0 for i in range(1, 21)]

    base_features = [
        float(race_config["base_lap_time"]),
        float(race_config["pit_lane_time"]),
        float(race_config["track_temp"]),
        float(total_laps),
        one_hot(start_tire, "SOFT"),
        one_hot(start_tire, "MEDIUM"),
        one_hot(start_tire, "HARD"),
        pit_count,
        laps_by_compound["SOFT"],
        laps_by_compound["MEDIUM"],
        laps_by_compound["HARD"],
        age_sum_by_compound["SOFT"],
        age_sum_by_compound["MEDIUM"],
        age_sum_by_compound["HARD"],
        age_sq_sum_by_compound["SOFT"],
        age_sq_sum_by_compound["MEDIUM"],
        age_sq_sum_by_compound["HARD"],
        one_hot(last_compound, "SOFT"),
        one_hot(last_compound, "MEDIUM"),
        one_hot(last_compound, "HARD"),
        stint_count,
        avg_pit_lap,
        first_pit_early,
    ]
    return base_features + driver_one_hot


def _load_linear_model(path: str) -> Tuple[List[float], float]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    weights = payload["weights"]
    if len(weights) != len(FEATURE_NAMES):
        raise ValueError("linear model feature count mismatch")
    return [float(x) for x in weights], float(payload.get("bias", 0.0))


def predict_with_linear_model(race_config: dict, strategies: dict) -> List[str]:
    """Predict finish order using learned linear weights, if available."""

    model_path = os.path.join(os.path.dirname(__file__), "linear_model.json")
    weights, bias = _load_linear_model(model_path)

    scored: List[Tuple[str, float]] = []
    for position in sorted(strategies.keys(), key=lambda k: int(k[3:])):
        strategy = strategies[position]
        driver_id = str(strategy["driver_id"])
        features = build_feature_vector(race_config, strategy)
        score = bias
        for value, weight in zip(features, weights):
            score += value * weight
        scored.append((driver_id, score))

    scored.sort(key=lambda item: (item[1], item[0]))
    return [driver_id for driver_id, _ in scored]


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


def _load_match_model(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, dict):
        raise ValueError("match model payload is not a dict")

    tables = payload.get("tables", {})
    if not isinstance(tables, dict):
        raise ValueError("match model missing tables")

    return payload


def _load_pair_match_model(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if not isinstance(payload, dict):
        raise ValueError("pair match model payload is not a dict")

    tables = payload.get("tables", {})
    if not isinstance(tables, dict):
        raise ValueError("pair match model missing tables")

    return payload


def _lookup_table_mean(model: Dict[str, Any], table_name: str, key: str) -> Tuple[float, int] | None:
    tables = model.get("tables", {})
    table = tables.get(table_name)
    if not isinstance(table, dict):
        return None
    value = table.get(key)
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        mean_pos = float(value[0])
        count = int(value[1])
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    return (mean_pos, count)


def _match_expected_positions(race_config: dict, strategies: dict, model: Dict[str, Any]) -> Dict[str, float]:
    """Estimate expected finish position per driver from matching backoff tables."""

    table_weights = {
        "exact": 4.0,
        "driver_strategy": 3.0,
        "coarse": 2.0,
        "driver_race": 1.5,
        "driver": 1.0,
    }

    global_mean = float(model.get("global_mean_pos", 10.5))
    expected_positions: Dict[str, float] = {}

    for position in sorted(strategies.keys(), key=lambda key: int(key[3:])):
        strategy = strategies[position]
        driver_id = str(strategy["driver_id"])
        keys = _build_match_keys(race_config, strategy)

        weighted_sum = 0.0
        weighted_count = 0.0
        for table_name, base_weight in table_weights.items():
            lookup = _lookup_table_mean(model, table_name, keys[table_name])
            if lookup is None:
                continue
            mean_pos, count = lookup
            weight = base_weight * math.log1p(float(count))
            weighted_sum += weight * mean_pos
            weighted_count += weight

        if weighted_count <= 0.0:
            expected_positions[driver_id] = global_mean
        else:
            expected_positions[driver_id] = weighted_sum / weighted_count

    return expected_positions


def _apply_match_model_correction(
    baseline_order: List[str],
    expected_positions: Dict[str, float],
    top_k: int = 12,
    max_swaps: int = 8,
    min_margin: float = 0.35,
) -> List[str]:
    """Apply conservative local swaps using match-model expected positions."""

    if not baseline_order:
        return baseline_order

    corrected = list(baseline_order)
    capped_top_k = max(2, min(len(corrected), int(top_k)))
    capped_swaps = max(0, int(max_swaps))
    margin = float(min_margin)

    swap_count = 0
    while swap_count < capped_swaps:
        changed = False
        for idx in range(capped_top_k - 1):
            first_driver = corrected[idx]
            second_driver = corrected[idx + 1]
            first_pos = expected_positions.get(first_driver, 10.5)
            second_pos = expected_positions.get(second_driver, 10.5)

            # Lower expected position is better; swap only when the match model is decisive.
            if (first_pos - second_pos) >= margin:
                corrected[idx], corrected[idx + 1] = corrected[idx + 1], corrected[idx]
                swap_count += 1
                changed = True
                if swap_count >= capped_swaps:
                    break
        if not changed:
            break

    return corrected


def predict_with_match_model(race_config: dict, strategies: dict, model: Dict[str, Any]) -> List[str]:
    """Predict finish order using data-driven matching with conservative correction."""

    baseline_order = simulate_race(
        race_config=race_config,
        strategies=strategies,
        rank_model=None,
    )
    expected_positions = _match_expected_positions(race_config, strategies, model)
    return _apply_match_model_correction(baseline_order, expected_positions)


def _lookup_pair_rate(model: Dict[str, Any], table_name: str, key: str) -> Tuple[float, int] | None:
    tables = model.get("tables", {})
    table = tables.get(table_name)
    if not isinstance(table, dict):
        return None
    value = table.get(key)
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        left_win_rate = float(value[0])
        count = int(value[1])
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    return (left_win_rate, count)


def _pair_driver_rate(model: Dict[str, Any], race_coarse: str, first_driver: str, second_driver: str) -> Tuple[float, int] | None:
    left, right = sorted((first_driver, second_driver))
    key = f"{race_coarse}|{left}|{right}"
    lookup = _lookup_pair_rate(model, "pair_driver_race", key)
    if lookup is None:
        return None
    left_rate, count = lookup
    if first_driver == left:
        return (left_rate, count)
    return (1.0 - left_rate, count)


def _pair_strategy_rate(
    model: Dict[str, Any],
    race_coarse: str,
    first_strategy_sig: str,
    second_strategy_sig: str,
) -> Tuple[float, int] | None:
    left, right = sorted((first_strategy_sig, second_strategy_sig))
    key = f"{race_coarse}|{left}|{right}"
    lookup = _lookup_pair_rate(model, "pair_strategy_race", key)
    if lookup is None:
        return None
    left_rate, count = lookup
    if first_strategy_sig == left:
        return (left_rate, count)
    return (1.0 - left_rate, count)


def _pair_driver_strategy_rate(
    model: Dict[str, Any],
    first_driver: str,
    first_strategy_sig: str,
    second_driver: str,
    second_strategy_sig: str,
) -> Tuple[float, int] | None:
    first_pair = f"{first_driver}|{first_strategy_sig}"
    second_pair = f"{second_driver}|{second_strategy_sig}"
    left, right = sorted((first_pair, second_pair))
    key = f"{left}|{right}"
    lookup = _lookup_pair_rate(model, "pair_driver_strategy", key)
    if lookup is None:
        return None
    left_rate, count = lookup
    if first_pair == left:
        return (left_rate, count)
    return (1.0 - left_rate, count)


def _pair_bias_score(
    model: Dict[str, Any],
    race_config: dict,
    strategies_by_driver: Dict[str, dict],
    first_driver: str,
    second_driver: str,
) -> float:
    race_coarse = _race_signature(race_config, coarse=True)
    total_laps = int(race_config["total_laps"])
    first_strategy = strategies_by_driver[first_driver]
    second_strategy = strategies_by_driver[second_driver]
    first_sig = _strategy_signature(first_strategy, total_laps=total_laps, coarse=True)
    second_sig = _strategy_signature(second_strategy, total_laps=total_laps, coarse=True)

    # Positive score => keep first ahead. Negative => second likely ahead.
    score = 0.0

    driver_lookup = _pair_driver_rate(model, race_coarse, first_driver, second_driver)
    if driver_lookup is not None:
        win_rate, count = driver_lookup
        score += 2.4 * (2.0 * win_rate - 1.0) * math.log1p(float(count))

    strategy_lookup = _pair_strategy_rate(model, race_coarse, first_sig, second_sig)
    if strategy_lookup is not None:
        win_rate, count = strategy_lookup
        score += 1.6 * (2.0 * win_rate - 1.0) * math.log1p(float(count))

    driver_strategy_lookup = _pair_driver_strategy_rate(
        model,
        first_driver,
        first_sig,
        second_driver,
        second_sig,
    )
    if driver_strategy_lookup is not None:
        win_rate, count = driver_strategy_lookup
        score += 2.0 * (2.0 * win_rate - 1.0) * math.log1p(float(count))

    return score


def predict_with_pair_match_model(race_config: dict, strategies: dict, model: Dict[str, Any]) -> List[str]:
    """Predict using pairwise data-driven matching as conservative swaps over physics."""

    baseline_order = simulate_race(
        race_config=race_config,
        strategies=strategies,
        rank_model=None,
    )

    strategies_by_driver: Dict[str, dict] = {}
    for position in sorted(strategies.keys(), key=lambda key: int(key[3:])):
        strategy = strategies[position]
        strategies_by_driver[str(strategy["driver_id"])] = strategy

    corrected = list(baseline_order)
    top_k = max(2, min(len(corrected), int(model.get("top_k", 12))))
    max_swaps = max(0, int(model.get("max_swaps", 8)))
    min_bias = float(model.get("min_bias", 0.55))

    swap_count = 0
    while swap_count < max_swaps:
        changed = False
        for idx in range(top_k - 1):
            first_driver = corrected[idx]
            second_driver = corrected[idx + 1]
            bias = _pair_bias_score(
                model=model,
                race_config=race_config,
                strategies_by_driver=strategies_by_driver,
                first_driver=first_driver,
                second_driver=second_driver,
            )
            if bias <= -min_bias:
                corrected[idx], corrected[idx + 1] = corrected[idx + 1], corrected[idx]
                swap_count += 1
                changed = True
                if swap_count >= max_swaps:
                    break
        if not changed:
            break

    return corrected


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


def _dot(values: List[float], weights: Tuple[float, ...]) -> float:
    total = 0.0
    for value, weight in zip(values, weights):
        total += value * weight
    return total


def _normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    if not raw_scores:
        return {}
    values = list(raw_scores.values())
    mean = sum(values) / float(len(values))
    variance = sum((v - mean) * (v - mean) for v in values) / float(len(values))
    stddev = math.sqrt(variance)
    if stddev < 1e-9:
        return {driver_id: 0.0 for driver_id in raw_scores}
    return {driver_id: (value - mean) / stddev for driver_id, value in raw_scores.items()}


def _apply_rank_model_correction(
    totals: List[Tuple[str, float]],
    race_config: dict,
    strategies: dict,
    average_pit_count: float,
    rank_model: RankCorrectionModel | None,
) -> List[Tuple[str, float]]:
    if rank_model is None or not totals:
        return totals

    if tuple(rank_model.feature_names) != FEATURE_NAMES:
        return totals

    total_laps = int(race_config["total_laps"])
    track_temp = float(race_config["track_temp"])
    regime_key = _regime_key(total_laps, track_temp, average_pit_count)
    weights = rank_model.regime_weights.get(regime_key, rank_model.global_weights)

    regime_knobs = None
    if rank_model.regime_inference:
        regime_knobs = rank_model.regime_inference.get(regime_key)

    alpha_seconds = float(regime_knobs.get("alpha_seconds", rank_model.alpha_seconds)) if regime_knobs else rank_model.alpha_seconds
    top_k_knob = int(regime_knobs.get("top_k", rank_model.top_k)) if regime_knobs else rank_model.top_k
    max_swaps_knob = int(regime_knobs.get("max_swaps", rank_model.max_swaps)) if regime_knobs else rank_model.max_swaps
    max_gap_knob = float(regime_knobs.get("max_gap_seconds", rank_model.max_gap_seconds)) if regime_knobs else rank_model.max_gap_seconds
    min_margin_knob = float(regime_knobs.get("min_margin", rank_model.min_margin)) if regime_knobs else rank_model.min_margin

    raw_rank_scores: Dict[str, float] = {}
    for position in sorted(strategies.keys(), key=lambda key: int(key[3:])):
        strategy = strategies[position]
        driver_id = str(strategy["driver_id"])
        features = build_feature_vector(race_config, strategy)
        raw_rank_scores[driver_id] = _dot(features, weights)

    normalized_scores = _normalize_scores(raw_rank_scores)

    adjusted: List[Tuple[str, float, float]] = []
    for driver_id, total_time in totals:
        score = normalized_scores.get(driver_id, 0.0)
        adjusted_time = total_time - alpha_seconds * score
        adjusted.append((driver_id, total_time, adjusted_time))

    adjusted.sort(key=lambda item: (item[2], item[0]))

    top_k = max(2, min(len(adjusted), int(top_k_knob)))
    max_swaps = max(0, int(max_swaps_knob))
    max_gap = float(max_gap_knob)
    min_margin = float(min_margin_knob)

    swap_count = 0
    while swap_count < max_swaps:
        changed = False
        for idx in range(top_k - 1):
            a_driver, _a_base, a_adjusted = adjusted[idx]
            b_driver, _b_base, b_adjusted = adjusted[idx + 1]
            if (b_adjusted - a_adjusted) > max_gap:
                continue

            margin = normalized_scores.get(b_driver, 0.0) - normalized_scores.get(a_driver, 0.0)
            if margin <= min_margin:
                continue

            adjusted[idx], adjusted[idx + 1] = adjusted[idx + 1], adjusted[idx]
            swap_count += 1
            changed = True
            if swap_count >= max_swaps:
                break
        if not changed:
            break

    return [(driver_id, adjusted_time) for driver_id, _base_time, adjusted_time in adjusted]


def _is_short_midtemp_pitlow(total_laps: int, track_temp: float, average_pit_count: float) -> bool:
    return total_laps < 48 and 25.0 <= track_temp <= 35.0 and average_pit_count < 1.5


def _apply_short_regime_rank_calibration(
    totals: List[Tuple[str, float]],
    calibrator: Dict[str, Any] | None,
    is_short_midtemp_pitlow: bool,
) -> List[Tuple[str, float]]:
    if not calibrator or not is_short_midtemp_pitlow:
        return totals

    if not bool(calibrator.get("enabled", False)):
        return totals

    target_regime = str(calibrator.get("target_regime", "temp_mid|laps_short|pit_low"))
    if target_regime != "temp_mid|laps_short|pit_low":
        return totals

    pair_biases = calibrator.get("pair_biases")
    if not isinstance(pair_biases, dict) or not pair_biases:
        return totals

    top_k = max(2, min(len(totals), int(calibrator.get("top_k", 8))))
    max_swaps = max(0, int(calibrator.get("max_swaps", 6)))
    max_gap = float(calibrator.get("max_gap", 0.12))
    min_pair_bias = float(calibrator.get("min_pair_bias", 0.22))

    ranked = list(totals)
    swap_count = 0
    while swap_count < max_swaps:
        changed = False
        for idx in range(top_k - 1):
            a_driver, a_time = ranked[idx]
            b_driver, b_time = ranked[idx + 1]
            if (b_time - a_time) > max_gap:
                continue

            pair_key = "|".join(sorted((a_driver, b_driver)))
            raw_bias = pair_biases.get(pair_key)
            if raw_bias is None:
                continue

            try:
                bias = float(raw_bias)
            except (TypeError, ValueError):
                continue
            if abs(bias) < min_pair_bias:
                continue

            left, right = pair_key.split("|")
            desired_first = left if bias > 0.0 else right
            if desired_first == b_driver:
                ranked[idx], ranked[idx + 1] = ranked[idx + 1], ranked[idx]
                swap_count += 1
                changed = True
                if swap_count >= max_swaps:
                    break
        if not changed:
            break

    return ranked


def simulate_race(
    race_config: dict,
    strategies: dict,
    params: ModelParams | None = None,
    rank_model: RankCorrectionModel | None = None,
) -> List[str]:
    """Simulate one race and return driver IDs ordered by finish position."""

    model = params if params is not None else load_model_params()

    total_laps = int(race_config["total_laps"])
    base_lap_time = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])

    average_pit_count = sum(
        len(strategies[f"pos{i}"].get("pit_stops", []))
        for i in range(1, 21)
    ) / 20.0
    is_short_midtemp_pitlow = _is_short_midtemp_pitlow(total_laps, track_temp, average_pit_count)
    effective_pit_lane_time = pit_lane_time
    if is_short_midtemp_pitlow:
        effective_pit_lane_time = pit_lane_time * model.short_race_pit_lane_scale

    totals: List[Tuple[str, float]] = []

    for position in sorted(strategies.keys(), key=lambda k: int(k[3:])):
        strategy = strategies[position]
        driver_id = str(strategy["driver_id"])
        current_tire = str(strategy["starting_tire"]).upper()
        pit_schedule = _build_pit_schedule(strategy.get("pit_stops", []))
        tire_age = 0
        total_time = 0.0
        is_outlap = False

        if model.driver_offsets:
            total_time += float(model.driver_offsets.get(driver_id, 0.0))
        if model.driver_temp_coeffs:
            total_time += float(model.driver_temp_coeffs.get(driver_id, 0.0)) * (track_temp - model.temp_reference)

        if is_short_midtemp_pitlow:
            if current_tire == "SOFT":
                total_time += model.short_race_start_soft_bias
            elif current_tire == "HARD":
                total_time += model.short_race_start_hard_bias

        for lap in range(1, total_laps + 1):
            tire_age += 1
            lap_adjustment = 0.0
            if lap in pit_schedule:
                lap_adjustment += model.pre_pit_lap_offset
            if is_outlap:
                lap_adjustment += model.post_pit_lap_offset
                is_outlap = False
            total_time += _lap_time(
                base_lap_time=base_lap_time,
                track_temp=track_temp,
                tire=current_tire,
                tire_age=tire_age,
                params=model,
                lap_adjustment=lap_adjustment,
            )

            if lap in pit_schedule:
                total_time += effective_pit_lane_time
                current_tire = pit_schedule[lap]
                tire_age = 0
                is_outlap = True

        totals.append((driver_id, total_time))

    totals.sort(key=lambda item: (item[1], item[0]))
    totals = _apply_rank_model_correction(
        totals=totals,
        race_config=race_config,
        strategies=strategies,
        average_pit_count=average_pit_count,
        rank_model=rank_model,
    )
    totals = _apply_short_regime_rank_calibration(
        totals=totals,
        calibrator=model.short_regime_rank_calibrator,
        is_short_midtemp_pitlow=is_short_midtemp_pitlow,
    )
    return [driver_id for driver_id, _ in totals]


def main() -> None:
    test_case = json.load(sys.stdin)
    race_id = test_case["race_id"]
    linear_model_path = os.path.join(os.path.dirname(__file__), "linear_model.json")
    match_model_path = os.path.join(os.path.dirname(__file__), "match_model.json")
    pair_match_model_path = os.path.join(os.path.dirname(__file__), "pair_match_model.json")
    use_linear_model = os.environ.get("BOXBOXBOX_USE_LINEAR", "").strip() == "1"
    use_match_model = os.environ.get("BOXBOXBOX_USE_MATCH", "").strip() == "1"
    use_pair_match_model = os.environ.get("BOXBOXBOX_USE_PAIR_MATCH", "").strip() == "1"
    use_rank_model = os.environ.get("BOXBOXBOX_USE_RANK", "1").strip() != "0"
    rank_model = load_rank_model() if use_rank_model else None

    if use_pair_match_model and os.path.exists(pair_match_model_path):
        try:
            pair_match_model = _load_pair_match_model(pair_match_model_path)
            finishing_positions = predict_with_pair_match_model(
                race_config=test_case["race_config"],
                strategies=test_case["strategies"],
                model=pair_match_model,
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            finishing_positions = simulate_race(
                race_config=test_case["race_config"],
                strategies=test_case["strategies"],
                rank_model=rank_model,
            )
    elif use_match_model and os.path.exists(match_model_path):
        try:
            match_model = _load_match_model(match_model_path)
            finishing_positions = predict_with_match_model(
                race_config=test_case["race_config"],
                strategies=test_case["strategies"],
                model=match_model,
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            finishing_positions = simulate_race(
                race_config=test_case["race_config"],
                strategies=test_case["strategies"],
                rank_model=rank_model,
            )
    elif use_linear_model and os.path.exists(linear_model_path):
        finishing_positions = predict_with_linear_model(
            race_config=test_case["race_config"],
            strategies=test_case["strategies"],
        )
    else:
        finishing_positions = simulate_race(
            race_config=test_case["race_config"],
            strategies=test_case["strategies"],
            rank_model=rank_model,
        )

    output = {
        "race_id": race_id,
        "finishing_positions": finishing_positions,
    }
    print(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    main()
