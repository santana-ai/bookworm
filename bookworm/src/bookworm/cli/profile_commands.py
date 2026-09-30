"""``filter-actor-speeches``, ``generate-profiles``, ``validate-profiles`` and the review."""

from pathlib import Path
from typing import Annotated

import typer

from bookworm.cli.common import (
    ProfilesConfigOption,
    ProfileValidationConfigOption,
    echo_json,
    echo_progress,
    exit_on_problems,
    guarded,
)
from bookworm.profiles.config import load_split_filter_config
from bookworm.profiles.generate import ClientFactory, GenerateRequest, run_generate_profiles
from bookworm.profiles.review import run_sample_profile_review, run_score_profile_review
from bookworm.profiles.split_filter import build_split_filter
from bookworm.profiles.validate import run_validate_profiles


def add_profile_commands(app: typer.Typer, client_factory: ClientFactory) -> None:
    @app.command(
        "filter-actor-speeches",
        help="Keep only the hearings of the configured splits in the actor speeches file.",
    )
    def filter_actor_speeches_command(
        config_path: ProfilesConfigOption,
        output: Annotated[
            Path | None,
            typer.Option("--output", help="Filtered speeches JSONL (overrides config)."),
        ] = None,
    ) -> None:
        stats = guarded(
            lambda: build_split_filter(load_split_filter_config(config_path).with_output(output))
        )
        echo_json(stats)

    @app.command(
        "generate-profiles",
        help="Write one LLM-written profile per actor from an actor speeches file.",
    )
    def generate_profiles_command(
        config_path: ProfilesConfigOption,
        speeches: Annotated[
            Path | None,
            typer.Option("--speeches", "--input", help="Actor speeches JSONL (overrides config)."),
        ] = None,
        output: Annotated[
            Path | None, typer.Option("--output", help="Profiles JSONL (overrides config).")
        ] = None,
        model: Annotated[
            str | None,
            typer.Option("--model", help="Hugging Face model id or local path (overrides config)."),
        ] = None,
        actors: Annotated[
            list[str] | None,
            typer.Option("--actors", help="Only these actors, by exact name; repeat for more."),
        ] = None,
        limit: Annotated[
            int | None, typer.Option("--limit", min=0, help="Process at most N actors this run.")
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run", help="Render every prompt and report sizes, without a model."
            ),
        ] = False,
    ) -> None:
        request = GenerateRequest(config_path, speeches, output, model, actors, limit, dry_run)
        outcome = guarded(lambda: run_generate_profiles(request, client_factory, echo_progress))
        echo_json(outcome.summary)
        exit_on_problems(outcome.ok)

    @app.command(
        "validate-profiles",
        help=(
            "Score each verified UDV against the profile of its actor and of every other "
            "actor, separating hearings seen at generation from held-out ones."
        ),
    )
    def validate_profiles_command(
        config_path: ProfileValidationConfigOption,
        overwrite: Annotated[
            bool, typer.Option("--overwrite", help="Replace existing pairs and report files.")
        ] = False,
    ) -> None:
        echo_json(guarded(lambda: run_validate_profiles(config_path, overwrite=overwrite)))

    @app.command(
        "sample-profile-review",
        help="Draw a seeded, stratified sample of profile pairs as a CSV with empty judgments.",
    )
    def sample_profile_review_command(
        config_path: ProfileValidationConfigOption,
        overwrite: Annotated[
            bool, typer.Option("--overwrite", help="Replace an existing review sample.")
        ] = False,
    ) -> None:
        echo_json(guarded(lambda: run_sample_profile_review(config_path, overwrite=overwrite)))

    @app.command(
        "score-profile-review",
        help="Compute support proportions with Wilson intervals from a filled review CSV.",
    )
    def score_profile_review_command(
        config_path: ProfileValidationConfigOption,
        annotations: Annotated[
            Path, typer.Option("--annotations", help="Review CSV with the judgments filled in.")
        ],
    ) -> None:
        echo_json(guarded(lambda: run_score_profile_review(config_path, annotations)))
