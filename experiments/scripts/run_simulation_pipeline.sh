#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MODEL="${MODEL:?set MODEL to a transformers model id; the udv_v2 profile and simulation run uses mlx-community/Qwen3.8-27B-8bit through experiments.mlx (see ../docs/methodology/mlx_backend.md)}"

scripts/run_actor_profiles.sh "$@"
uv run python -m experiments.actors.evaluate_simulation --model "$MODEL"
uv run python -m experiments.actors.simulate --model "$MODEL"
