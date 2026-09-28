"""The console summary of a ``score-annotation`` report."""

from typing import Any

from experiments.validation.precision_report import SUPPORT_QUESTION

Record = dict[str, Any]

RUNS = ("udv_v1", "udv_v2_unchanged", "udv_v2")


def interval_text(interval: Record) -> str:
    return f"{interval['estimate']} [{interval['low']}, {interval['high']}]"


def print_stratum(name: str, result: Record) -> None:
    judged = f"    {name:26s} judged {result['judged_of_sample']:9s} "
    if result["question"] == SUPPORT_QUESTION:
        strict, tolerant = result["strict_precision"], result["tolerant_precision"]
        print(
            f"{judged}strict={strict['estimate']} [{strict['low']}, {strict['high']}] "
            f"tolerant={tolerant['estimate']} [{tolerant['low']}, {tolerant['high']}]"
        )
    else:
        rate = result["persons"]["false_absence_rate"]
        print(f"{judged}false_absence(persons)={rate['estimate']} [{rate['low']}, {rate['high']}]")


def print_score_summary(report: Record) -> None:
    print(f"{report['sample_name']}: {report['status'].upper()}, judged {report['judged']}")
    for label in report["invalid_labels"]:
        print(f"  not counted, label outside the allowed set: {label['item_id']}")
    for run in [name for name in RUNS if name in report]:
        print(f"  {run}")
        for name, result in report[run]["strata"].items():
            print_stratum(name, result)
        for rule in report[run]["criteria"]:
            print(
                f"    criterion {rule['name']}: {rule['status']} "
                f"(lower {rule['observed']['low']} vs {rule['min_wilson_lower']})"
            )
    if "udv_v2" in report:
        print_combined_extras(report["udv_v2"])


def print_sensitivity(sensitivity: Record) -> None:
    print(
        f"    sensitivity: {sensitivity['superset_left_out']} of "
        f"{sensitivity['superset_judged']} judged superset items left out "
        "(udv_v1 label not correta)"
    )
    inherited = sensitivity["superset_inherited"]["by_stratum"]
    for name, kept in sensitivity["superset_only_if_v1_correta"]["by_stratum"].items():
        print(
            f"    {name:26s} strict {interval_text(inherited[name]['strict_precision'])} "
            f"inherited vs {interval_text(kept['strict_precision'])} correta only; "
            f"tolerant {interval_text(inherited[name]['tolerant_precision'])} vs "
            f"{interval_text(kept['tolerant_precision'])}"
        )


def print_combined_extras(combined: Record) -> None:
    for relation, counts in combined["sources"].items():
        print(
            f"    source {relation:12s} items={counts['items']} judged={counts['judged']} "
            f"label from the {counts['label_from']}"
        )
    for tier, result in combined["by_tier"].items():
        print(
            f"    tier {tier:21s} n={result['judged_udvs']} "
            f"strict={interval_text(result['strict_precision'])} "
            f"tolerant={interval_text(result['tolerant_precision'])}"
        )
    print_sensitivity(combined["sensitivity"])
