#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_SPEECHES="artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl"
TRAIN_PROFILES="artifacts/actor_profiles/actor_profiles_train.jsonl"
MODEL="${MODEL:?set MODEL to a transformers model id; the udv_v2 profile and simulation run uses mlx-community/Qwen3.8-27B-8bit through experiments.mlx (see ../docs/methodology/mlx_backend.md)}"

if [ ! -f dataset/PublicHearingBR_LDS.jsonl ]; then
  uv run python -m experiments.data.download
fi

uv run python -m experiments.actors.build_speeches
uv run python -m experiments.actors.filter_speeches --output "$TRAIN_SPEECHES"
uv run python -m experiments.actors.generate_profiles \
  --model "$MODEL" \
  --input "$TRAIN_SPEECHES" \
  --output "$TRAIN_PROFILES" \
  "$@"
