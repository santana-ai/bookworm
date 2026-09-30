"""Wilson intervals and how many successes a pre-declared criterion needs at a given size."""

from typing import Any

from scipy.stats import binomtest

from experiments.validation.generate_sample.config import ValidationConfig

Record = dict[str, Any]


def wilson_interval(successes: int, trials: int, confidence_level: float) -> Record:
    if trials == 0:
        return {"successes": 0, "trials": 0, "estimate": None, "low": None, "high": None}
    interval = binomtest(successes, trials).proportion_ci(
        confidence_level=confidence_level, method="wilson"
    )
    return {
        "successes": successes,
        "trials": trials,
        "estimate": successes / trials,
        "low": float(interval.low),
        "high": float(interval.high),
    }


def min_successes_to_pass(trials: int, min_lower: float, confidence_level: float) -> int | None:
    for successes in range(trials + 1):
        low = wilson_interval(successes, trials, confidence_level)["low"]
        if low is not None and low >= min_lower:
            return successes
    return None


def criteria_feasibility(
    config: ValidationConfig, trials_by_stratum: dict[str, int]
) -> list[Record]:
    rows = []
    for rule in config.source["criteria"]["rules"]:
        trials = trials_by_stratum.get(rule["stratum"], 0)
        needed = min_successes_to_pass(trials, rule["min_wilson_lower"], config.confidence_level)
        rows.append(
            {
                "rule": rule["name"],
                "stratum": rule["stratum"],
                "trials": trials,
                "min_successes_to_pass": needed,
                "max_failures_to_pass": None if needed is None else trials - needed,
            }
        )
    return rows
