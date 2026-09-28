#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: MODEL=<model id> scripts/run_simulation_pipeline.sh [generate_profiles options]

Runs scripts/run_actor_profiles.sh, then the multiple-choice evaluation
(experiments.actors.evaluate_simulation) and the open generation (experiments.actors.simulate)
with the same model. MODEL is a transformers model id and is required; the udv_v2 profile and
simulation run uses mlx-community/Qwen3.8-27B-8bit through experiments.mlx (see
../docs/methodology/mlx_backend.md). Other options are passed to generate_profiles.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ -z "${MODEL:-}" ]]; then
  usage >&2
  exit 1
fi
export MODEL

cd "$(dirname "$0")/.."

scripts/run_actor_profiles.sh "$@"
uv run python -m experiments.actors.evaluate_simulation --model "$MODEL"
uv run python -m experiments.actors.simulate --model "$MODEL"
