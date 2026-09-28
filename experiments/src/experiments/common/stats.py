import zlib
from typing import Any

import numpy as np
from scipy.stats import binomtest

Record = dict[str, Any]


def stream_rng(seed: int, label: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(label.encode())])


def hearing_sums(values: np.ndarray, hearing_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(hearing_ids, return_inverse=True)
    groups = int(inverse.max()) + 1 if len(inverse) else 0
    sums = np.bincount(inverse, weights=values, minlength=groups)
    counts = np.bincount(inverse, minlength=groups).astype(np.float64)
    return sums, counts


def bootstrap_mean(
    values: np.ndarray,
    hearing_ids: np.ndarray,
    samples: int,
    level: float,
    rng: np.random.Generator,
) -> Record:
    if len(values) == 0:
        return {"point": None, "low": None, "high": None, "replicates": 0}
    sums, counts = hearing_sums(values.astype(np.float64), hearing_ids)
    draws = rng.integers(0, len(sums), size=(samples, len(sums)))
    replicates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - level) / 2
    return {
        "point": float(values.mean()),
        "low": float(np.quantile(replicates, tail)),
        "high": float(np.quantile(replicates, 1 - tail)),
        "replicates": samples,
    }


def mcnemar_exact(baseline: np.ndarray, other: np.ndarray) -> Record:
    only_baseline = int((baseline & ~other).sum())
    only_other = int((~baseline & other).sum())
    discordant = only_baseline + only_other
    p_value = 1.0 if discordant == 0 else binomtest(only_other, discordant, 0.5).pvalue
    return {
        "only_baseline_correct": only_baseline,
        "only_other_correct": only_other,
        "discordant": discordant,
        "p_value": float(p_value),
    }


def sign_flip_test(
    differences: np.ndarray, hearing_ids: np.ndarray, samples: int, rng: np.random.Generator
) -> Record:
    sums, _ = hearing_sums(differences.astype(np.float64), hearing_ids)
    observed = float(sums.sum())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(sums)))
    null = signs @ sums
    extreme = int((np.abs(null) >= abs(observed) - 1e-12).sum())
    return {
        "statistic": "sum over hearings of the per-hearing sum of differences",
        "observed": observed,
        "samples": samples,
        "p_value": (extreme + 1) / (samples + 1),
    }


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values))
    running = 0.0
    total = len(p_values)
    for position, index in enumerate(order):
        running = max(running, min(1.0, (total - position) * p_values[index]))
        adjusted[index] = running
    return [float(value) for value in adjusted]
