import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaModel,
    T5ForTokenClassification,
)

from utils.grounding_scorers import (
    ALIGNSCORE_IGNORED_KEYS,
    HHEM_PROMPT,
    MINICHECK_LABEL_IDS,
    CandidateSpec,
    Pair,
    PairScore,
    Record,
    batched,
    length_order,
    softmax_rows,
    spec_signature,
    two_way_probability,
)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def snapshot(name: str, revision: str, patterns: list[str] | None = None) -> Path:
    return Path(
        snapshot_download(name, revision=revision, allow_patterns=patterns, local_files_only=True)
    )


def token_lengths(tokenizer: Any, texts: list[str], pairs: list[str] | None = None) -> list[int]:
    encoded = tokenizer(texts, pairs, add_special_tokens=True, truncation=False)
    return [len(ids) for ids in encoded["input_ids"]]


class TorchScorer:
    def __init__(self, spec: CandidateSpec, device: str) -> None:
        self.spec = spec
        self.device = device
        self.signature = spec_signature(spec, device)
        self.dtype = DTYPES[spec.dtype]
        self.info: Record = {
            "name": spec.name,
            "revision": spec.revision,
            "device": device,
            "dtype": spec.dtype,
            "max_length": spec.max_length,
        }
        started = time.perf_counter()
        self.load()
        self.info["load_seconds"] = round(time.perf_counter() - started, 3)

    def load(self) -> None:
        raise NotImplementedError

    def batch_values(self, pairs: list[Pair]) -> tuple[np.ndarray, list[int], list[Record]]:
        raise NotImplementedError

    def score(self, pairs: Sequence[Pair]) -> list[PairScore]:
        items = list(pairs)
        results: list[PairScore | None] = [None] * len(items)
        for batch in batched(length_order(items), self.spec.batch_size):
            with torch.inference_mode():
                values, lengths, extras = self.batch_values([items[i] for i in batch])
            for position, index in enumerate(batch):
                results[index] = PairScore(
                    value=float(values[position]),
                    tokens=int(lengths[position]),
                    truncated=bool(lengths[position] > self.spec.max_length),
                    extra=extras[position],
                )
        return [result for result in results if result is not None]

    def to_device(self, encoded: Any) -> dict[str, torch.Tensor]:
        return {name: tensor.to(self.device) for name, tensor in encoded.items()}


class MiniCheckT5Scorer(TorchScorer):
    def load(self) -> None:
        path = snapshot(self.spec.name, self.spec.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=self.dtype)
        self.model.to(self.device).eval()
        self.info["label_ids"] = list(MINICHECK_LABEL_IDS)
        self.info["label_tokens"] = self.tokenizer.convert_ids_to_tokens(list(MINICHECK_LABEL_IDS))

    def texts(self, pairs: list[Pair]) -> list[str]:
        eos = self.tokenizer.eos_token
        return [f"predict: {premise}{eos}{hypothesis}" for premise, hypothesis in pairs]

    def batch_values(self, pairs: list[Pair]) -> tuple[np.ndarray, list[int], list[Record]]:
        texts = self.texts(pairs)
        lengths = token_lengths(self.tokenizer, texts)
        encoded = self.tokenizer(
            texts,
            max_length=self.spec.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = self.to_device(encoded)
        decoder = torch.zeros((len(texts), 1), dtype=torch.long, device=self.device)
        logits = self.model(**inputs, decoder_input_ids=decoder).logits[:, 0, :]
        array = logits.float().cpu().numpy()
        negative, positive = MINICHECK_LABEL_IDS
        values = two_way_probability(array, positive, negative)
        return values, lengths, [{} for _ in pairs]


class MiniCheckEncoderScorer(TorchScorer):
    def load(self) -> None:
        path = snapshot(self.spec.name, self.spec.revision, self.spec.raw.get("files"))
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        config = AutoConfig.from_pretrained(path, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, config=config, dtype=self.dtype
        )
        self.model.to(self.device).eval()
        self.info["eos_token"] = self.tokenizer.eos_token

    def batch_values(self, pairs: list[Pair]) -> tuple[np.ndarray, list[int], list[Record]]:
        eos = self.tokenizer.eos_token
        texts = [f"{premise}{eos}{hypothesis}" for premise, hypothesis in pairs]
        lengths = token_lengths(self.tokenizer, texts)
        encoded = self.tokenizer(
            texts,
            max_length=self.spec.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        logits = self.model(**self.to_device(encoded)).logits.float().cpu().numpy()
        return softmax_rows(logits)[:, 1], lengths, [{} for _ in pairs]


class FactCgScorer(MiniCheckEncoderScorer):
    def batch_values(self, pairs: list[Pair]) -> tuple[np.ndarray, list[int], list[Record]]:
        template = self.spec.raw["template"]
        texts = [
            template.format(premise=premise, hypothesis=hypothesis) for premise, hypothesis in pairs
        ]
        lengths = token_lengths(self.tokenizer, texts)
        encoded = self.tokenizer(
            texts,
            max_length=self.spec.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        logits = self.model(**self.to_device(encoded)).logits.float().cpu().numpy()
        return softmax_rows(logits)[:, 1], lengths, [{} for _ in pairs]


class AlignScoreHead(torch.nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.base_model = RobertaModel(config, add_pooling_layer=True)
        self.tri_layer = torch.nn.Linear(config.hidden_size, 3)

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.base_model(**inputs).pooler_output
        return self.tri_layer(pooled)


class AlignScoreScorer(TorchScorer):
    def load(self) -> None:
        raw = self.spec.raw
        backbone = snapshot(raw["backbone"], raw["backbone_revision"], raw["backbone_files"])
        checkpoint = hf_hub_download(
            self.spec.name,
            raw["checkpoint_file"],
            revision=self.spec.revision,
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        head = AlignScoreHead(AutoConfig.from_pretrained(backbone))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        wanted = {
            key: value
            for key, value in state.items()
            if key.startswith(("base_model.", "tri_layer."))
            and not key.endswith(ALIGNSCORE_IGNORED_KEYS)
        }
        missing, unexpected = head.load_state_dict(wanted, strict=False)
        if missing or unexpected:
            raise SystemExit(f"AlignScore keys: missing {missing}, unexpected {unexpected}")
        self.model = head.to(self.device, dtype=self.dtype).eval()
        self.info["checkpoint_file"] = raw["checkpoint_file"]
        self.info["loaded_keys"] = len(wanted)
        self.info["unused_prefixes"] = sorted(
            {key.split(".")[0] for key in state if key not in wanted}
        )

    def batch_values(self, pairs: list[Pair]) -> tuple[np.ndarray, list[int], list[Record]]:
        premises = [premise for premise, _ in pairs]
        hypotheses = [hypothesis for _, hypothesis in pairs]
        lengths = token_lengths(self.tokenizer, premises, hypotheses)
        encoded = self.tokenizer(
            premises,
            hypotheses,
            max_length=self.spec.max_length,
            truncation="only_first",
            padding=True,
            return_tensors="pt",
        )
        logits = self.model(**self.to_device(encoded)).float().cpu().numpy()
        return softmax_rows(logits)[:, 0], lengths, [{} for _ in pairs]


class HhemScorer(TorchScorer):
    def load(self) -> None:
        raw = self.spec.raw
        foundation = snapshot(
            raw["foundation"], raw["foundation_revision"], raw["foundation_files"]
        )
        weights = snapshot(self.spec.name, self.spec.revision) / "model.safetensors"
        self.tokenizer = AutoTokenizer.from_pretrained(foundation)
        config = AutoConfig.from_pretrained(foundation, num_labels=2)
        model = T5ForTokenClassification(config)
        state = {
            key.removeprefix("t5."): value
            for key, value in load_file(str(weights)).items()
            if key.startswith("t5.")
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        tied = {"transformer.encoder.embed_tokens.weight"}
        if set(missing) - tied or unexpected:
            raise SystemExit(f"HHEM keys: missing {missing}, unexpected {unexpected}")
        self.model = cast(Any, model).to(self.device, dtype=self.dtype).eval()
        self.info["missing_tied_keys"] = sorted(missing)

    def batch_values(self, pairs: list[Pair]) -> tuple[np.ndarray, list[int], list[Record]]:
        texts = [
            HHEM_PROMPT.format(text1=premise, text2=hypothesis) for premise, hypothesis in pairs
        ]
        lengths = token_lengths(self.tokenizer, texts)
        encoded = self.tokenizer(
            texts,
            max_length=self.spec.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        logits = self.model(**self.to_device(encoded)).logits[:, 0, :].float().cpu().numpy()
        return softmax_rows(logits)[:, 1], lengths, [{} for _ in pairs]


class RerankerScorer(TorchScorer):
    def load(self) -> None:
        path = snapshot(self.spec.name, self.spec.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=self.dtype)
        self.model.to(self.device).eval()

    def batch_values(self, pairs: list[Pair]) -> tuple[np.ndarray, list[int], list[Record]]:
        queries = [hypothesis for _, hypothesis in pairs]
        passages = [premise for premise, _ in pairs]
        lengths = token_lengths(self.tokenizer, queries, passages)
        encoded = self.tokenizer(
            queries,
            passages,
            max_length=self.spec.max_length,
            truncation="only_second",
            padding=True,
            return_tensors="pt",
        )
        logits = self.model(**self.to_device(encoded)).logits[:, 0].float().cpu().numpy()
        return logits, lengths, [{} for _ in pairs]


class YesNoScorer(TorchScorer):
    positive_word = "Yes"
    negative_word = "No"
    template_options: Record = {}

    def load(self) -> None:
        path = snapshot(self.spec.name, self.spec.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=self.dtype)
        cast(Any, self.model).to(self.device).eval()
        self.yes_id = self.first_token(self.positive_word)
        self.no_id = self.first_token(self.negative_word)
        if self.yes_id == self.no_id:
            raise SystemExit("the Yes and No answers start with the same token")
        self.info["answer_tokens"] = {
            self.positive_word: [self.yes_id, self.tokenizer.convert_ids_to_tokens(self.yes_id)],
            self.negative_word: [self.no_id, self.tokenizer.convert_ids_to_tokens(self.no_id)],
        }
        self.info["template_options"] = self.template_options

    def first_token(self, word: str) -> int:
        return int(self.tokenizer.encode(word, add_special_tokens=False)[0])

    def prompt(self, premise: str, hypothesis: str) -> str:
        raise NotImplementedError

    def positive_value(self, yes: np.ndarray) -> np.ndarray:
        return yes

    def batch_values(self, pairs: list[Pair]) -> tuple[np.ndarray, list[int], list[Record]]:
        texts = [self.prompt(premise, hypothesis) for premise, hypothesis in pairs]
        encoded = self.tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")
        lengths = [int(n) for n in encoded["attention_mask"].sum(dim=1).tolist()]
        if max(lengths) > self.spec.max_length:
            raise SystemExit(f"{self.spec.key}: a prompt has {max(lengths)} tokens")
        output = self.model(**self.to_device(encoded), logits_to_keep=1)
        logits = output.logits[:, -1, :].float().cpu().numpy()
        full = softmax_rows(logits)
        yes = two_way_probability(logits, self.yes_id, self.no_id)
        mass = full[:, self.yes_id] + full[:, self.no_id]
        extras = [{"answer_mass": round(float(value), 6)} for value in mass]
        return self.positive_value(yes), lengths, extras


class LlmJudgeScorer(YesNoScorer):
    def load(self) -> None:
        if "Qwen3-" in self.spec.name and "Instruct" not in self.spec.name:
            self.template_options = {"enable_thinking": False}
        super().load()

    def prompt(self, premise: str, hypothesis: str) -> str:
        content = self.spec.raw["prompt"].format(premise=premise, hypothesis=hypothesis)
        messages = [{"role": "user", "content": content}]
        return cast(
            str,
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.template_options
            ),
        )


class GraniteGuardianScorer(YesNoScorer):
    def prompt(self, premise: str, hypothesis: str) -> str:
        messages = [
            {"role": "context", "content": premise},
            {"role": "assistant", "content": hypothesis},
        ]
        return cast(
            str,
            self.tokenizer.apply_chat_template(
                messages,
                guardian_config={"risk_name": "groundedness"},
                tokenize=False,
                add_generation_prompt=True,
            ),
        )

    def positive_value(self, yes: np.ndarray) -> np.ndarray:
        return 1.0 - yes


SCORER_CLASSES: dict[str, type[TorchScorer]] = {
    "minicheck_t5": MiniCheckT5Scorer,
    "minicheck_encoder": MiniCheckEncoderScorer,
    "factcg": FactCgScorer,
    "alignscore": AlignScoreScorer,
    "hhem": HhemScorer,
    "reranker": RerankerScorer,
    "llm_judge": LlmJudgeScorer,
    "granite_guardian": GraniteGuardianScorer,
}


def load_scorer(spec: CandidateSpec, device: str) -> TorchScorer:
    return SCORER_CLASSES[spec.kind](spec, device)


def release(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
