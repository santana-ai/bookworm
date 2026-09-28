"""The explore command: cross-validate every candidate on train and select one."""

import argparse
import json
from typing import Any

from bookworm import sha256_of_file

from experiments.udv.calibrate_threshold import rounded
from experiments.verifier.exploration.candidates import (
    check_reading,
    fixed_candidates,
    learned_candidates,
    reference_candidate,
    resolve_reference,
)
from experiments.verifier.exploration.config import KIND_ORDER, ExplorationConfig
from experiments.verifier.exploration.cross_validation import (
    cv_folds,
    evaluate_fixed,
    evaluate_learned,
    select,
    summary_row,
)
from experiments.verifier.exploration.provenance import code_hashes
from experiments.verifier.exploration.scores import (
    candidate_scores,
    load_all,
    output_dir,
    write_csv,
)
from experiments.verifier.nli.provenance import environment
from experiments.verifier.runtime import now

Record = dict[str, Any]


def command_explore(args: argparse.Namespace, config: ExplorationConfig) -> None:
    split = config.raw["fit_split"]
    ids, labels, hearings, data, missing = load_all(config, split)
    reading = check_reading(config, data)
    folds = cv_folds(config, labels, hearings)
    rows = []
    fixed = fixed_candidates(config, data)
    for candidate in fixed:
        aucs, kappas = evaluate_fixed(candidate_scores(candidate, data), labels, folds)
        rows.append(summary_row(candidate, aucs, kappas, config.n_splits))
    print(f"{len(fixed)} fixed candidates evaluated on {len(folds)} folds", flush=True)
    learned_c: Record = {}
    for candidate in learned_candidates(config, data):
        aucs, kappas, chosen = evaluate_learned(candidate, data, labels, hearings, folds, config)
        rows.append(summary_row(candidate, aucs, kappas, config.n_splits))
        learned_c[candidate.key] = {str(c): chosen.count(c) for c in config.c_grid}
        print(f"evaluated {candidate.key}", flush=True)
    references = {}
    for reference in config.raw["selection"]["references"]:
        aucs, kappas = evaluate_fixed(resolve_reference(reference, config, data), labels, folds)
        references[reference] = {
            k: rounded(v) if isinstance(v, float) else v
            for k, v in summary_row(
                reference_candidate(reference), aucs, kappas, config.n_splits
            ).items()
        }
    best, selected, eligible = select(rows)
    rows.sort(key=lambda row: -row["cv_roc_auc_mean"])
    out = output_dir(config, args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "cv_results.csv", rows)
    selection = {
        "experiment": "nli_verifier_exploration",
        "name": config.name,
        "created_at": now(),
        "split_read": split,
        "files_read": {
            key: {"path": str(s.path), "sha256": sha256_of_file(s.path)} for key, s in data.items()
        },
        "scorers_missing": missing,
        "opinions": {
            "count": len(ids),
            "hearings": int(len(set(hearings.tolist()))),
            "not_inferable": int((~labels).sum()),
        },
        "folds": len(folds),
        "candidates": {
            "total": len(rows),
            "by_kind": {k: sum(r["kind"] == k for r in rows) for k in KIND_ORDER},
        },
        "reading_check": reading,
        "learned_c_chosen_per_fold": learned_c,
        "references_cv": references,
        "best": {k: rounded(v) if isinstance(v, float) else v for k, v in best.items()},
        "eligible_count": len(eligible),
        "selected": {k: rounded(v) if isinstance(v, float) else v for k, v in selected.items()},
        "selection_rule": config.raw["selection"]["rule"],
        "config": {"path": str(config.path), "sha256": sha256_of_file(config.path)},
        "code": code_hashes(),
        "environment": environment(),
    }
    with open(out / "selection.json", "w") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "best": selection["best"],
                "selected": selection["selected"],
                "references_cv": references,
            },
            indent=2,
        )
    )
