#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: MODEL=<model id> scripts/run_actor_profiles.sh [generate_profiles options]

Builds the per-actor speech files, keeps their train hearings and writes one profile per
actor with experiments.actors.generate_profiles. MODEL is a transformers model id and is
required; the udv_v2 profile and simulation run uses mlx-community/Qwen3.8-27B-8bit through
experiments.mlx (see ../docs/methodology/mlx_backend.md). Other options, such as --actors or
--limit, are passed to generate_profiles.
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

cd "$(dirname "$0")/.."

readonly LDS_FILE="dataset/PublicHearingBR_LDS.jsonl"
readonly TRAIN_SPEECHES="artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl"
readonly TRAIN_PROFILES="artifacts/actor_profiles/actor_profiles_train.jsonl"

if [[ ! -f "$LDS_FILE" ]]; then
  uv run python -m experiments.data.download
fi

uv run python -m experiments.actors.build_speeches
uv run python -m experiments.actors.filter_speeches --output "$TRAIN_SPEECHES"
uv run python -m experiments.actors.generate_profiles \
  --model "$MODEL" \
  --input "$TRAIN_SPEECHES" \
  --output "$TRAIN_PROFILES" \
  "$@"
