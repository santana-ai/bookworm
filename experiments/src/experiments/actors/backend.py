"""Transformers backend: the chat client that writes profiles and the model that scores and
generates for the actor simulation. `experiments.mlx.backend` provides the same two classes
for MLX."""

from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from experiments.actors.chat import (
    ChatResult,
    Generation,
    check_output_budget,
    letter_token_ids,
    strip_think,
    system_user_messages,
)

if TYPE_CHECKING:
    from experiments.actors.generate_profiles import ProfilesConfig

Record = dict[str, Any]


class TransformersModel:
    """A causal language model and its tokenizer, loaded with from_pretrained."""

    def __init__(self, name: str, device_map: str) -> None:
        self.tokenizer: Any = AutoTokenizer.from_pretrained(name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(name, device_map=device_map)

    def encode(self, messages: list[Record]) -> Any:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)


class TransformersChatClient(TransformersModel):
    def __init__(self, config: "ProfilesConfig") -> None:
        super().__init__(config.model, config.device_map)
        self._config = config

    def chat(self, system: str, user: str) -> ChatResult:
        config = self._config
        inputs = self.encode(system_user_messages(system, user))
        if config.seed is not None:
            set_seed(config.seed)
        output = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_output_tokens,
        )
        input_tokens = inputs["input_ids"].shape[1]
        generated = output[0, input_tokens:]
        check_output_budget(len(generated), config.max_output_tokens)
        content = self.tokenizer.decode(generated, skip_special_tokens=True)
        return ChatResult(
            text=strip_think(content),
            input_tokens=input_tokens,
            output_tokens=len(generated),
        )


class SimulationModel(TransformersModel):
    def __init__(self, name: str, device_map: str) -> None:
        super().__init__(name, device_map)
        self.letter_ids = letter_token_ids(self.tokenizer)

    @torch.inference_mode()
    def letter_logprobs(self, messages: list[Record]) -> list[float]:
        logits = self.model(**self.encode(messages)).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1)[self.letter_ids].tolist()

    @torch.inference_mode()
    def generate(self, messages: list[Record], max_new_tokens: int) -> Generation:
        inputs = self.encode(messages)
        output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        generated = output[0, inputs["input_ids"].shape[1] :]
        return Generation(
            text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            output_tokens=len(generated),
            truncated=len(generated) >= max_new_tokens,
        )
