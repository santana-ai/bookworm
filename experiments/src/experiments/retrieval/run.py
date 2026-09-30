"""The retrieval harness commands: ``run`` scores queries, ``summarize`` tests, ``queue`` chains."""

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from bookworm import write_json

from experiments.common.hub_offline import enforce_offline, offline_environment, offline_state
from experiments.common.udv_run import load_config as load_udv_config
from experiments.common.udv_run import seed_everything, select_device
from experiments.retrieval.config import (
    ExperimentConfig,
    RunPlan,
    SummaryPlan,
    load_config,
    parse_list,
    resolve_choice,
    resolve_retrievers,
    resolve_splits,
)
from experiments.retrieval.data import BENCHES, UNIT_KINDS
from experiments.retrieval.models import Runtime
from experiments.retrieval.reports import (
    RunContext,
    code_hashes,
    run_report,
    run_report_name,
    summary_report,
    summary_report_name,
)
from experiments.retrieval.scoring import run_retriever, write_query_rows
from experiments.retrieval.summary import load_query_rows, print_summary, summarize_groups
from experiments.retrieval.workload import Workload, build_workload

ENTRY_MODULE = "experiments.retrieval.experiments"
DEFAULT_CONFIG = Path("configs/retrieval_experiments.toml")
SMOKE_SUFFIX = "_smoke"


def run_name_for(args: argparse.Namespace, config: ExperimentConfig) -> str:
    if args.run_name:
        return str(args.run_name)
    limited = args.limit_queries is not None or args.limit_hearings is not None
    return f"{config.run['run_name']}{SMOKE_SUFFIX}" if limited else config.run["run_name"]


def run_plan(args: argparse.Namespace, config: ExperimentConfig) -> RunPlan:
    run_name = run_name_for(args, config)
    return RunPlan(
        run_name=run_name,
        run_dir=Path(config.run["output_dir"]) / run_name,
        splits=resolve_splits(parse_list(args.splits), args.final_test, config),
        retrievers=resolve_retrievers(parse_list(args.retrievers), config),
        kinds=resolve_choice(parse_list(args.units), config.unit_kinds, "--units"),
        benches=resolve_choice(parse_list(args.benches), tuple(config.benches), "--benches"),
        final_test=args.final_test,
        limit_hearings=args.limit_hearings,
        limit_queries=args.limit_queries,
    )


def build_runtime(config: ExperimentConfig, device: str, workload: Workload) -> Runtime:
    return Runtime(
        device=device,
        embeddings_dir=Path(config.cache["embeddings_dir"]),
        rerank_dir=Path(config.cache["rerank_dir"]),
        shard_size=config.cache["shard_size"],
        udv_config=load_udv_config(Path(config.cache["production_config"])),
        masked_texts_by_hearing=workload.masked_texts_by_hearing,
    )


def print_workload(workload: Workload, plan: RunPlan, device: str, load_seconds: float) -> None:
    total = sum(len(queries) for queries in workload.queries.values())
    print(
        f"{len(workload.hearings)} hearings, {total} queries ({', '.join(plan.splits)}) | "
        f"retrievers {plan.retrievers} | units {list(plan.kinds)} | device {device} | "
        f"loaded in {load_seconds:.1f}s",
        flush=True,
    )


def command_run(args: argparse.Namespace, config: ExperimentConfig) -> None:
    plan = run_plan(args, config)
    code = code_hashes()
    offline = enforce_offline() if config.run["hf_hub_offline"] else offline_state()
    seed_everything(config.run["seed"])
    device = select_device(args.device or config.run["device"])
    started = time.perf_counter()
    workload = build_workload(
        config, plan.splits, plan.benches, plan.limit_hearings, plan.limit_queries
    )
    context = RunContext(device, offline, code, time.perf_counter() - started)
    print_workload(workload, plan, device, context.load_seconds)
    runtime = build_runtime(config, device, workload)
    for retriever_id in plan.retrievers:
        rows, details = run_retriever(retriever_id, workload, plan.kinds, config, runtime)
        written = write_query_rows(rows, plan.run_dir, retriever_id, plan.splits)
        report = run_report(config, plan, context, workload, retriever_id, details, written)
        write_json(report, plan.run_dir / "runs" / run_report_name(plan, retriever_id))
        print(f"[{retriever_id}] {len(rows)} rows, {details['timing']}", flush=True)


def summary_plan(args: argparse.Namespace, config: ExperimentConfig) -> SummaryPlan:
    run_name = args.run_name or config.run["run_name"]
    return SummaryPlan(
        run_name=run_name,
        run_dir=Path(config.run["output_dir"]) / run_name,
        splits=resolve_splits(parse_list(args.splits), args.final_test, config),
        final_test=args.final_test,
        baseline_unit=args.baseline_unit or config.evaluation["baseline_unit"],
        baseline_retriever=args.baseline_retriever or config.evaluation["baseline_retriever"],
    )


def command_summarize(args: argparse.Namespace, config: ExperimentConfig) -> None:
    plan = summary_plan(args, config)
    code = code_hashes()
    grouped, inputs = load_query_rows(plan.run_dir, plan.splits)
    if not grouped:
        raise SystemExit(f"no query rows for {plan.splits} under {plan.run_dir}")
    summaries, comparisons = summarize_groups(
        grouped, plan.splits, plan.baseline, config.evaluation
    )
    report = summary_report(config, plan, summaries, comparisons, inputs, code)
    name = summary_report_name(plan, report["baseline"]["configured"])
    write_json(report, plan.run_dir / name)
    print_summary(summaries)


def shared_options(args: argparse.Namespace) -> list[str]:
    options = ["--config", str(args.config)]
    if args.run_name:
        options += ["--run-name", args.run_name]
    if args.splits:
        options += ["--splits", args.splits]
    if args.final_test:
        options += ["--final-test"]
    return options


def queue_commands(args: argparse.Namespace, config: ExperimentConfig) -> list[list[str]]:
    base = [sys.executable, "-m", ENTRY_MODULE]
    shared = shared_options(args)
    commands = []
    for step in config.queue_steps:
        command = [*base, "run", *shared, "--retrievers", ",".join(step["retrievers"])]
        for key in ("units", "benches"):
            if step.get(key):
                command += [f"--{key}", ",".join(step[key])]
        if step.get("device"):
            command += ["--device", step["device"]]
        commands.append(command)
    commands.append([*base, "summarize", *shared])
    return commands


def command_queue(args: argparse.Namespace, config: ExperimentConfig) -> None:
    commands = queue_commands(args, config)
    for number, command in enumerate(commands):
        print(f"[{number}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    env = offline_environment() if config.run["hf_hub_offline"] else None
    for number, command in enumerate(commands):
        if number < args.from_step:
            continue
        print(f"[queue] step {number}: {shlex.join(command)}", flush=True)
        started = time.perf_counter()
        subprocess.run(command, check=True, env=env)
        print(f"[queue] step {number} done in {time.perf_counter() - started:.0f}s", flush=True)


COMMANDS = {"run": command_run, "summarize": command_summarize, "queue": command_queue}


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--splits", default=None, help="comma list; default from the config")
    parser.add_argument("--final-test", action="store_true", help="allow the test split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval harness: rank the speaker's candidate units for each opinion "
        "(masked-quote and NLI benchmarks) with sparse, dense, hybrid and reranking retrievers."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="score queries and write per-query rows")
    add_shared(run)
    run.add_argument("--retrievers", default=None, help="comma list of ids or base names")
    run.add_argument("--units", default=None, help=f"comma list among {UNIT_KINDS}")
    run.add_argument("--benches", default=None, help=f"comma list among {BENCHES}")
    run.add_argument("--limit-queries", type=int, default=None, help="first N queries per bench")
    run.add_argument("--limit-hearings", type=int, default=None, help="first N hearings")
    run.add_argument("--device", default=None, help="override run.device (auto, mps, cpu)")
    summarize = commands.add_parser("summarize", help="metrics, intervals and paired tests")
    add_shared(summarize)
    summarize.add_argument("--baseline-retriever", default=None, help="default from the config")
    summarize.add_argument("--baseline-unit", default=None, help="default from the config")
    queue = commands.add_parser("queue", help="run every configured step, one model per process")
    add_shared(queue)
    queue.add_argument("--dry-run", action="store_true", help="only print the commands")
    queue.add_argument("--from-step", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    COMMANDS[args.command](args, load_config(args.config))


if __name__ == "__main__":
    main()
