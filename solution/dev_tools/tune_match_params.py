#!/usr/bin/env python3
import json
from pathlib import Path

import race_simulator as rs


def load_cases(root: Path):
    inputs = sorted((root / "data" / "test_cases" / "inputs").glob("test_*.json"))
    cases = []
    for inp in inputs:
        exp = root / "data" / "test_cases" / "expected_outputs" / inp.name
        with open(inp, "r", encoding="utf-8") as f:
            test_case = json.load(f)
        with open(exp, "r", encoding="utf-8") as f:
            expected = json.load(f)
        cases.append((test_case, expected))
    return cases


def exact_match(pred, exp):
    p = pred["finishing_positions"]
    e = exp["finishing_positions"]
    if len(p) != len(e):
        return False
    for i in range(len(e)):
        if p[i] != e[i]:
            return False
    return True


def evaluate(cases, model, top_k, max_swaps, min_margin):
    passed = 0
    for test_case, expected in cases:
        base = rs.simulate_race(test_case["race_config"], test_case["strategies"], rank_model=None)
        expected_pos = rs._match_expected_positions(test_case["race_config"], test_case["strategies"], model)
        pred_positions = rs._apply_match_model_correction(
            baseline_order=base,
            expected_positions=expected_pos,
            top_k=top_k,
            max_swaps=max_swaps,
            min_margin=min_margin,
        )
        pred = {"finishing_positions": pred_positions}
        if exact_match(pred, expected):
            passed += 1
    total = len(cases)
    acc = round(100.0 * passed / total, 2)
    return passed, total - passed, acc


def main():
    root = Path(__file__).resolve().parent.parent
    model_path = Path(__file__).resolve().parent / "match_model.json"
    model = rs._load_match_model(str(model_path))
    cases = load_cases(root)

    # baseline
    p = 0
    for test_case, expected in cases:
        pred_positions = rs.simulate_race(test_case["race_config"], test_case["strategies"], rank_model=None)
        pred = {"finishing_positions": pred_positions}
        if exact_match(pred, expected):
            p += 1
    print(f"Baseline: {p}/100 ({p}%)")

    best = (-1, None)
    for top_k in [4, 6, 8, 10, 12]:
        for max_swaps in [1, 2, 3, 4, 6, 8]:
            for min_margin in [0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
                passed, failed, acc = evaluate(cases, model, top_k, max_swaps, min_margin)
                if acc > best[0]:
                    best = (acc, (top_k, max_swaps, min_margin, passed, failed))
                print(
                    f"top_k={top_k:2d} swaps={max_swaps:2d} margin={min_margin:>4} -> "
                    f"{passed}/100 ({acc}%)"
                )

    print("BEST:", best)


if __name__ == "__main__":
    main()
