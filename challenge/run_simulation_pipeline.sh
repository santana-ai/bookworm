#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"

./run_actor_profiles.sh "$@"
uv run python -m utils.evaluate_actor_simulation --model "$MODEL"
uv run python -m utils.simulate_actors --model "$MODEL"
