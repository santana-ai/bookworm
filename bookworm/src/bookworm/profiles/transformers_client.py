from collections.abc import Callable
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from bookworm.profiles.config import ModelSettings
from bookworm.profiles.llm import ChatResult, finish_generation

ModelLoader = Callable[[ModelSettings], tuple[Any, Any]]


def load_model(settings: ModelSettings) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(settings.name)
    model = AutoModelForCausalLM.from_pretrained(settings.name, device_map=settings.device_map)
    return tokenizer, model


class TransformersChatClient:
    def __init__(self, settings: ModelSettings, *, loader: ModelLoader = load_model) -> None:
        self._settings = settings
        self._tokenizer, self._model = loader(settings)

    def chat(self, system: str, user: str) -> ChatResult:
        inputs = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._model.device)
        if self._settings.seed is not None:
            set_seed(self._settings.seed)
        output = self._model.generate(
            **inputs,
            do_sample=True,
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            max_new_tokens=self._settings.max_output_tokens,
        )
        input_tokens = int(inputs["input_ids"].shape[1])
        generated = output[0, input_tokens:]
        return finish_generation(
            self._tokenizer.decode(generated, skip_special_tokens=True),
            input_tokens,
            len(generated),
            self._settings.max_output_tokens,
        )
