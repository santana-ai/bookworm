"""The ``bookworm`` Typer application."""

from typing import Annotated

import typer

from bookworm import __version__
from bookworm.cli.export_commands import add_export_commands
from bookworm.cli.profile_commands import add_profile_commands
from bookworm.cli.split_commands import add_split_commands
from bookworm.cli.udv_commands import EncoderFactory, add_udv_commands, default_encoder_factory
from bookworm.profiles.generate import ClientFactory, default_client_factory

APP_HELP = (
    "Build and verify evidence units (UDVs), temporal splits and actor profiles of the "
    "PublicHearingBR hearings."
)


def show_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def create_app(
    encoder_factory: EncoderFactory = default_encoder_factory,
    client_factory: ClientFactory = default_client_factory,
) -> typer.Typer:
    """Build the Typer application; the factories let tests run commands without a model."""
    app = typer.Typer(name="bookworm", help=APP_HELP, no_args_is_help=True, add_completion=False)

    @app.callback()
    def main(
        version: Annotated[
            bool,
            typer.Option(
                "--version",
                callback=show_version,
                is_eager=True,
                help="Show the version and exit.",
            ),
        ] = False,
    ) -> None:
        pass

    add_udv_commands(app, encoder_factory)
    add_export_commands(app)
    add_split_commands(app)
    add_profile_commands(app, client_factory)
    return app


app = create_app()
