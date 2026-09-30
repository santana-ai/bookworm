"""Scores for arbitrary premise and hypothesis pairs read from a JSONL file."""

import argparse
import json
from typing import Any

from bookworm import sha256_of_file, write_json, write_jsonl

from experiments.common.reporting import file_record
from experiments.common.splits import load_split_lookup
from experiments.verifier.benchmark_inputs import split_records
from experiments.verifier.nli.benchmark import load_pair_units
from experiments.verifier.nli.config import (
    DECISION_KINDS,
    ScorerSpec,
    VerifierConfig,
    scored_splits,
    selected_scorers,
)
from experiments.verifier.nli.decision import check_decision_scorer
from experiments.verifier.nli.scoring import device_of, score_report, score_units

Record = dict[str, Any]


def command_pairs(args: argparse.Namespace, config: VerifierConfig) -> None:
    splits = scored_splits(config, args.final_test)
    keys = args.scorers if args.scorers is not None else list(config.pairs_default_scorers)
    specs = selected_scorers(config, keys)
    split_of, split_source = load_split_lookup(config.manifest_path, config.lds_sha256)
    units, inputs = load_pair_units(args.input, split_of, splits)
    english = [spec.key for spec in specs if spec.language != "pt"]
    if english:
        raise SystemExit(f"pairs reads Portuguese propositions only; {english} have language en")
    for spec in specs:
        if spec.kind in DECISION_KINDS:
            check_decision_scorer(spec, config, args.decision_mode)
    device = device_of(args, config)
    out_dir = (args.output_dir or config.output_dir) / "pairs" / args.run_name
    context = {
        "sources": {
            "input": file_record(args.input),
            "splits": split_source,
        },
        "format": {
            "input": config.source["pairs"]["input_format"],
            "output": config.source["pairs"]["output_format"],
        },
        "premise": {"concat_separator": config.concat_separator},
        "subset": None,
    }
    by_id = {row["pair_id"]: row for row in inputs}
    print(f"{len(units)} pairs ({', '.join(splits)}) on {device} -> {out_dir}", flush=True)
    for spec in specs:
        rows, details = score_units(
            units,
            spec,
            config,
            device,
            False,
            f"pairs_{args.run_name}",
            decision_mode=args.decision_mode,
        )
        output = [pair_output_row(row, by_id[row["id"]], spec) for row in rows]
        path = out_dir / f"{spec.key}.jsonl"
        write_jsonl(output, path)
        files = {"pairs": {"path": str(path), "rows": len(output), "sha256": sha256_of_file(path)}}
        report = score_report(
            "pairs",
            args.run_name,
            spec,
            rows,
            details,
            context,
            split_records(splits, args.final_test),
            config,
            files,
        )
        write_json(report, out_dir / f"{spec.key}_report.json")
        print(
            f"{spec.key}: {len(output)} pairs, timing {json.dumps(details['timing'])}", flush=True
        )


def pair_output_row(row: Record, source: Record, spec: ScorerSpec) -> Record:
    header = {key: value for key, value in row.items() if key != "id"}
    return {
        "pair_id": row["id"],
        "query_id": source["query_id"],
        **header,
        "scorer": spec.key,
        "model": spec.name,
        "revision": spec.revision,
        "meta": source.get("meta"),
    }
