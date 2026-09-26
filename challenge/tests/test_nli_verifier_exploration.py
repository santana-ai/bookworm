from pathlib import Path

import numpy as np
import pytest

from utils.nli_verifier_exploration import (
    Candidate,
    ScorerData,
    candidate_scores,
    cv_folds,
    fixed_candidates,
    load_exploration_config,
    pool_values,
    select,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "nli_verifier_exploration.toml"


@pytest.fixture
def config(monkeypatch):
    monkeypatch.chdir(CONFIG.parents[1])
    return load_exploration_config(CONFIG)


def scorer(kind: str = "decision") -> ScorerData:
    return ScorerData(
        key="toy",
        kind=kind,
        path=Path("toy"),
        chunks={"a": [[0.2, 0.8, 0.5], [], [0.4]], "b": [[0.6, 0.0, 0.1], [], [1.0]]},
        concatenated={"a": np.array([0.7, np.nan, 0.3]), "b": np.array([0.1, np.nan, 0.9])},
        stored={},
    )


def test_pool_values():
    assert pool_values([0.2, 0.8, 0.5], "max") == 0.8
    assert pool_values([0.2, 0.8, 0.5], "mean") == pytest.approx(0.5)
    assert pool_values([0.2, 0.8, 0.5], "top2_mean") == pytest.approx(0.65)
    assert pool_values([0.4], "top2_mean") == pytest.approx(0.4)
    assert pool_values([0.5, 0.5], "noisy_or") == pytest.approx(0.75)
    assert pool_values([], "max") == 0.0
    assert pool_values([], "max", -1.0) == -1.0


def test_panel_is_pooled_after_the_per_chunk_mean():
    data = {"toy": scorer()}
    panel = Candidate("p", "panel", "toy", ("a", "b"), "max")
    assert candidate_scores(panel, data).tolist() == pytest.approx([0.4, 0.0, 0.7])
    joined = Candidate("c", "panel", "toy", ("a", "b"), "concatenated")
    assert candidate_scores(joined, data).tolist() == pytest.approx([0.4, 0.0, 0.6])


def test_empty_premise_uses_the_cosine_floor():
    data = {"toy": scorer("cosine")}
    signal = Candidate("s", "signal", "toy", ("a",), "mean")
    assert candidate_scores(signal, data)[1] == -1.0


def test_selection_prefers_the_simplest_candidate_within_one_se():
    rows = [
        {
            "candidate": "l",
            "kind": "learned",
            "component_count": 24,
            "cv_roc_auc_mean": 0.80,
            "cv_roc_auc_se": 0.02,
        },
        {
            "candidate": "p",
            "kind": "panel",
            "component_count": 3,
            "cv_roc_auc_mean": 0.79,
            "cv_roc_auc_se": 0.02,
        },
        {
            "candidate": "s",
            "kind": "signal",
            "component_count": 1,
            "cv_roc_auc_mean": 0.77,
            "cv_roc_auc_se": 0.02,
        },
    ]
    best, selected, eligible = select(rows)
    assert best["candidate"] == "l"
    assert selected["candidate"] == "p"
    assert {row["candidate"] for row in eligible} == {"l", "p"}


def test_folds_keep_each_hearing_on_one_side(config):
    rng = np.random.default_rng(0)
    hearings = np.repeat(np.arange(40), 5)
    labels = rng.random(len(hearings)) > 0.2
    folds = cv_folds(config, labels, hearings)
    assert len(folds) == config.n_splits * config.repeats
    for fit, held in folds:
        assert not set(hearings[fit]) & set(hearings[held])
        assert len(fit) + len(held) == len(hearings)


def test_fixed_candidates_cover_every_panel_subset(config):
    data = {"toy": scorer()}
    data["toy"].chunks = {name: [[0.5]] for name in config.decision_signals}
    data["toy"].concatenated = {name: np.array([0.5]) for name in config.decision_signals}
    candidates = fixed_candidates(config, data)
    panels = [c for c in candidates if c.kind == "panel"]
    subsets = 2 ** len(config.panel_components) - 1 - len(config.panel_components)
    assert len(panels) == subsets * len(config.pools)
    assert len([c for c in candidates if c.kind == "signal"]) == len(config.decision_signals) * len(
        config.pools
    )
