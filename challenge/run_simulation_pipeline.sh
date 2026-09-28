#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MODEL="${MODEL:?set MODEL to a transformers model id; the udv_v2 profile and simulation run uses mlx-community/Qwen3.8-27B-8bit through mlx_alternative (see mlx_alternative/README.md)}"

./run_actor_profiles.sh "$@"
uv run python -m utils.evaluate_actor_simulation --model "$MODEL"
uv run python -m utils.simulate_actors --model "$MODEL"
