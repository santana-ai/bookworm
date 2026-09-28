"""Pinned seq2seq translator and length-sorted batched translation of missing texts."""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedConfig

from experiments.common.hub_offline import pinned_weights_file
from experiments.common.transcript import (
    normalize_whitespace,
)
from experiments.verifier.runtime import now, throughput
from experiments.verifier.translate.config import DecodingSpec, ModelSpec
from experiments.verifier.translate.store import (
    TranslationOutput,
    TranslationStore,
    Translator,
    model_signature,
    signature_digest,
)

Record = dict[str, Any]


def print_progress(done: int, total: int, started: float) -> None:
    rate, remaining = throughput(done, total, started)
    print(f"  {done}/{total} texts, {rate:.2f}/s, about {remaining:.1f} min left", flush=True)


def length_batches(lengths: list[int], batch_size: int, token_budget: int = 0) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    batches: list[list[int]] = []
    for index in order:
        current = batches[-1] if batches else None
        if (
            current is not None
            and len(current) < batch_size
            and (not token_budget or (len(current) + 1) * lengths[current[0]] <= token_budget)
        ):
            current.append(index)
        else:
            batches.append([index])
    return batches


def translate_missing(
    translator: Translator,
    store: TranslationStore,
    texts: Iterable[str],
    progress_every: int = 0,
    token_budget: int = 0,
) -> Record:
    if signature_digest(translator.signature) != store.digest:
        raise SystemExit("the translator signature differs from the store signature")
    missing = store.missing(texts)
    lengths = translator.model_input_tokens(missing) if missing else []
    size = translator.batch_size
    started = time.perf_counter()
    batches = 0
    done = 0
    for batches, indices in enumerate(length_batches(lengths, size, token_budget), start=1):
        batch = [missing[index] for index in indices]
        batch_started = time.perf_counter()
        outputs = translator.translate(batch)
        provenance = {
            "device": translator.device,
            "batch_size": size,
            "batch_token_budget": token_budget,
            "batch_number": batches,
            "batch_texts": len(batch),
            "batch_seconds": time.perf_counter() - batch_started,
            "created_at": now(),
        }
        store.append(batch, outputs, provenance)
        done += len(batch)
        if progress_every and batches % progress_every == 0:
            print_progress(done, len(missing), started)
    seconds = time.perf_counter() - started
    return {
        "translated": len(missing),
        "batches": batches,
        "seconds": round(seconds, 2),
        "texts_per_second": round(len(missing) / seconds, 4) if seconds and missing else None,
        "model_input_tokens": int(sum(lengths)),
        "model_input_tokens_per_second": (
            round(sum(lengths) / seconds, 2) if seconds and missing else None
        ),
    }


@dataclass
class Seq2SeqTranslator:
    spec: ModelSpec
    decoding: DecodingSpec
    tokenizer: Any
    model: Any
    device: str
    signature: Record
    batch_size: int
    target_token_id: int
    special_tokens: int
    info: Record

    def model_input_tokens(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        encoded = self.tokenizer(texts, truncation=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def output_length(self, sequence: list[int]) -> tuple[int, bool]:
        body = sequence[1:]
        eos = self.tokenizer.eos_token_id
        finished = eos in body
        end = body.index(eos) if finished else len(body)
        forced = 1 if body and body[0] == self.target_token_id else 0
        return end - forced, finished

    def translate(self, texts: list[str]) -> list[TranslationOutput]:
        full = self.model_input_tokens(texts)
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.decoding.max_input_tokens,
            padding=True,
            return_tensors="pt",
        )
        longest = int(encoded["attention_mask"].sum(dim=1).max())
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded.to(self.device),
                forced_bos_token_id=self.target_token_id,
                max_new_tokens=self.decoding.max_new_tokens(longest),
                **self.decoding.generation_kwargs(),
            )
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs = []
        for count, sequence, text in zip(full, generated.cpu().tolist(), decoded, strict=True):
            length, finished = self.output_length(sequence)
            outputs.append(
                TranslationOutput(
                    text=normalize_whitespace(text),
                    input_tokens=count - self.special_tokens,
                    model_input_tokens=count,
                    output_tokens=length,
                    input_truncated=count > self.decoding.max_input_tokens,
                    hit_max_new_tokens=not finished,
                )
            )
        return outputs


def load_tokenizer(spec: ModelSpec) -> Any:
    return AutoTokenizer.from_pretrained(
        spec.name, revision=spec.revision, src_lang=spec.src_lang, local_files_only=True
    )


def check_tokenizer(tokenizer: Any, spec: ModelSpec) -> Record:
    source_id = tokenizer.convert_tokens_to_ids(spec.src_token)
    target_id = tokenizer.convert_tokens_to_ids(spec.tgt_token)
    if tokenizer.unk_token_id in (source_id, target_id):
        raise SystemExit(f"{spec.name}: {spec.src_token} or {spec.tgt_token} is unknown")
    probe = tokenizer("Bom dia a todos.")["input_ids"]
    if probe[0] != source_id or probe[-1] != tokenizer.eos_token_id:
        raise SystemExit(f"{spec.name}: inputs do not start with {spec.src_token} and end in </s>")
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "source_token_id": source_id,
        "target_token_id": target_id,
        "special_tokens_per_input": tokenizer.num_special_tokens_to_add(),
        "input_layout": "[src_token] tokens [</s>], checked on a probe sentence",
    }


def resolved_commit(spec: ModelSpec) -> str | None:
    config_dict, _ = PreTrainedConfig.get_config_dict(
        spec.name, revision=spec.revision, local_files_only=True
    )
    return config_dict.get("_commit_hash")


def load_translator(spec: ModelSpec, decoding: DecodingSpec, device: str) -> Seq2SeqTranslator:
    started = time.perf_counter()
    commit = resolved_commit(spec)
    if commit != spec.revision:
        raise SystemExit(f"{spec.name}: resolved {commit} != {spec.revision}")
    weights = pinned_weights_file(spec.name, spec.revision)
    if not weights["snapshot_is_pinned_revision"]:
        raise SystemExit(
            f"{spec.name}: weights file {weights['path']} is not in the pinned snapshot"
        )
    tokenizer = load_tokenizer(spec)
    tokenizer_info = check_tokenizer(tokenizer, spec)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        spec.name,
        revision=spec.revision,
        dtype=getattr(torch, spec.dtype),
        local_files_only=True,
        use_safetensors=weights["format"] == "safetensors",
    )
    model.to(device).eval()
    info = {
        "role": spec.role,
        "name": spec.name,
        "revision": spec.revision,
        "resolved_commit": commit,
        "license": spec.license,
        "weights_file": weights,
        "tokenizer": tokenizer_info,
        "dtype": str(next(model.parameters()).dtype),
        "device": device,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "torch_threads": torch.get_num_threads(),
        "load_seconds": round(time.perf_counter() - started, 1),
    }
    return Seq2SeqTranslator(
        spec=spec,
        decoding=decoding,
        tokenizer=tokenizer,
        model=model,
        device=device,
        signature=model_signature(spec, decoding),
        batch_size=decoding.batch_size,
        target_token_id=tokenizer_info["target_token_id"],
        special_tokens=tokenizer_info["special_tokens_per_input"],
        info=info,
    )
