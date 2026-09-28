import json
import os
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

from mlx_alternative.backend import BackendOptions, register_model
from utils import build_actor_speeches, filter_actor_speeches
from utils.download_dataset import download_public_hearing_br

Record = dict[str, Any]

MODELS_DIR_VARIABLE = "MLX_MODELS_DIR"
DEFAULT_SETTINGS = Path("mlx_alternative/config.yaml")
BASE_CONFIGS = Path("configs")
DATASET_DIR = Path("dataset")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str
    local_dir: Path
    description: str
    options: BackendOptions


@dataclass(frozen=True)
class Settings:
    runs_dir: Path
    models_dir: Path
    env_file: Path | None
    models: dict[str, ModelSpec]
    short_run_actors: tuple[str, ...]
    benchmark: Record


def load_settings(path: Path = DEFAULT_SETTINGS) -> Settings:
    with open(path) as f:
        raw = yaml.safe_load(f)
    backend = raw.get("backend", {})
    models_dir = Path(os.environ.get(MODELS_DIR_VARIABLE) or raw["models_dir"]).expanduser()
    models = {
        model_id: ModelSpec(
            id=model_id,
            repo=entry["repo"],
            local_dir=models_dir / entry["repo"],
            description=entry.get("description", ""),
            options=BackendOptions(
                template_kwargs=tuple(sorted((entry.get("chat_template_kwargs") or {}).items())),
                prefix_cache=bool(backend.get("prefix_cache", True)),
                prefill_step=int(backend.get("prefill_step", 2048)),
            ),
        )
        for model_id, entry in raw["models"].items()
    }
    return Settings(
        runs_dir=Path(raw["runs_dir"]),
        models_dir=models_dir,
        env_file=Path(raw["env_file"]) if raw.get("env_file") else None,
        models=models,
        short_run_actors=tuple(raw.get("short_run", {}).get("actors", [])),
        benchmark=raw.get("benchmark", {}),
    )


def model_spec(settings: Settings, model_id: str) -> ModelSpec:
    if model_id not in settings.models:
        raise SystemExit(f"unknown model {model_id!r}; choose among {sorted(settings.models)}")
    return settings.models[model_id]


def load_env_file(settings: Settings) -> None:
    if settings.env_file is None or not settings.env_file.exists():
        return
    for line in settings.env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def download_model(settings: Settings, spec: ModelSpec) -> Path:
    load_env_file(settings)
    snapshot_download(repo_id=spec.repo, local_dir=spec.local_dir)
    return spec.local_dir


def activate_model(spec: ModelSpec) -> None:
    if not (spec.local_dir / "config.json").exists():
        raise SystemExit(
            f"{spec.repo} not found in {spec.local_dir}; download it with"
            f" `python -m mlx_alternative.run download --model {spec.id}`, or set"
            f" {MODELS_DIR_VARIABLE} to the folder that holds <org>/<model>"
        )
    register_model(spec.repo, spec.local_dir)


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def toml_lines(data: Record, prefix: tuple[str, ...] = ()) -> list[str]:
    lines = [
        f"{key} = {toml_value(value)}"
        for key, value in data.items()
        if not isinstance(value, dict) and not is_table_array(value)
    ]
    for key, value in data.items():
        path = ".".join((*prefix, key))
        if isinstance(value, dict):
            lines += ["", f"[{path}]", *toml_lines(value, (*prefix, key))]
        elif is_table_array(value):
            for item in value:
                lines += ["", f"[[{path}]]", *toml_lines(item, (*prefix, key))]
    return lines


def write_toml(data: Record, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(toml_lines(data)).lstrip("\n") + "\n")
    return path


def read_base(name: str) -> Record:
    with open(BASE_CONFIGS / name, "rb") as f:
        return tomllib.load(f)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    shared: Path
    profiles: Path
    simulation_dir: Path
    timings: Path
    hearing_actors_config: Path
    profiles_config: Path
    simulation_config: Path


def run_paths(settings: Settings, spec: ModelSpec) -> RunPaths:
    root = settings.runs_dir / spec.id
    shared = settings.runs_dir / "shared"
    return RunPaths(
        root=root,
        shared=shared,
        profiles=root / "actor_profiles" / "actor_profiles_train.jsonl",
        simulation_dir=root / "actor_simulation",
        timings=root / "timings.jsonl",
        hearing_actors_config=shared / "configs" / "hearing_actors.toml",
        profiles_config=root / "configs" / "actor_profiles.toml",
        simulation_config=root / "configs" / "actor_simulation.toml",
    )


def write_derived_configs(spec: ModelSpec, paths: RunPaths) -> None:
    hearing_actors = read_base("hearing_actors.toml")
    hearing_actors["measurement"]["output_path"] = str(paths.shared / "measurements.json")
    hearing_actors["speeches"]["ambiguous_names_path"] = str(paths.shared / "ambiguous_names.json")
    hearing_actors["speeches"]["stats_path"] = str(paths.shared / "actor_speeches_stats.json")
    write_toml(hearing_actors, paths.hearing_actors_config)

    profiles = read_base("actor_profiles.toml")
    profiles["output"]["profiles_path"] = str(paths.profiles)
    profiles["split_filter"]["stats_path"] = str(paths.shared / "train_speeches_stats.json")
    profiles["model"]["name"] = spec.repo
    write_toml(profiles, paths.profiles_config)

    simulation = read_base("actor_simulation.toml")
    simulation["input"]["profiles_path"] = str(paths.profiles)
    simulation["output"]["dir"] = str(paths.simulation_dir)
    simulation["model"]["name"] = spec.repo
    write_toml(simulation, paths.simulation_config)


def call_main(module: str, main: Callable[[], None], argv: list[str]) -> None:
    saved = sys.argv
    sys.argv = [module, *argv]
    try:
        main()
    except SystemExit as exit_:
        if exit_.code not in (None, 0):
            raise
    finally:
        sys.argv = saved


def train_speeches_path(paths: RunPaths) -> Path:
    with open(paths.profiles_config, "rb") as f:
        return Path(tomllib.load(f)["split_filter"]["speeches_path"])


def prepare_speeches(paths: RunPaths) -> None:
    with open(paths.profiles_config, "rb") as f:
        profiles = tomllib.load(f)
    with open(paths.hearing_actors_config, "rb") as f:
        hearing_actors = tomllib.load(f)
    if not Path(profiles["input"]["lds_path"]).exists():
        download_public_hearing_br(DATASET_DIR)
    if not Path(hearing_actors["speeches"]["multi_hearing_path"]).exists():
        call_main(
            "utils.build_actor_speeches",
            build_actor_speeches.main,
            ["--config", str(paths.hearing_actors_config)],
        )
    if not Path(profiles["split_filter"]["speeches_path"]).exists():
        call_main(
            "utils.filter_actor_speeches",
            filter_actor_speeches.main,
            ["--config", str(paths.profiles_config)],
        )


def append_timing(paths: RunPaths, entry: Record) -> None:
    paths.timings.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.timings, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_timings(paths: RunPaths) -> list[Record]:
    if not paths.timings.exists():
        return []
    with open(paths.timings) as f:
        return [json.loads(line) for line in f if line.strip()]
