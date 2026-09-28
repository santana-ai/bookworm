#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TRAIN_SPEECHES="artifacts/cache/hearing_actors/actors_multi_hearing_train.jsonl"
TRAIN_PROFILES="artifacts/actor_profiles/actor_profiles_train.jsonl"
MODEL="${MODEL:?set MODEL to a transformers model id; the udv_v2 profile and simulation run uses mlx-community/Qwen3.8-27B-8bit through mlx_alternative (see mlx_alternative/README.md)}"

if [ ! -f dataset/PublicHearingBR_LDS.jsonl ]; then
  uv run python -m utils.download_dataset
fi

uv run python -m utils.build_actor_speeches
uv run python -m utils.filter_actor_speeches --output "$TRAIN_SPEECHES"
uv run python -m utils.generate_actor_profiles \
  --model "$MODEL" \
  --input "$TRAIN_SPEECHES" \
  --output "$TRAIN_PROFILES" \
  "$@"
