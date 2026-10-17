"""Run an open vision-language model in this process with Transformers.

For a single GPU box or Colab, where starting a model server is one moving part
too many. The checkpoint is loaded with ``trust_remote_code=False``, so only
architectures that Transformers ships (Qwen3-VL, Qwen2.5-VL, InternVL3 among
them) run, never code downloaded with the weights. It implements the same
``generate(images, prompt) -> text`` contract as ``OpenAICompatibleClient``,
plus ``batch`` so the views of a building share one forward pass, ``chat`` and
``stream`` for the viewer's assistant.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .vlm import MAX_ANSWER_TOKENS, Message, user_message

# Apache-2.0, about 9 GB in fp16: fits a 16 GB T4 next to the analysis models.
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DTYPES = ("auto", "float16", "bfloat16", "float32")


def _resolve_dtype(torch: Any, name: str, device: str) -> Any:
    if name != "auto":
        return getattr(torch, name)
    if device.startswith("cuda"):
        # T4 and older GPUs have no bfloat16 units.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Half the memory of float32 on CPU, and the weights are published in bfloat16.
    return torch.bfloat16


class TransformersVLM:
    def __init__(
        self, model_id: str = DEFAULT_MODEL, *, cache_dir: Path | None = None, revision: str | None = None,
        device: str | None = None, dtype: str = "auto", max_new_tokens: int = MAX_ANSWER_TOKENS, batch_size: int = 4,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {', '.join(DTYPES)}")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id, self.max_new_tokens, self.batch_size = model_id, max_new_tokens, max(1, batch_size)
        options = {"cache_dir": str(cache_dir) if cache_dir else None, "revision": revision, "trust_remote_code": False}
        self.processor = AutoProcessor.from_pretrained(model_id, **options)
        # Generation appends to the right, so batched prompts are padded on the left.
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=_resolve_dtype(torch, dtype, self.device), **options)
        self.model.to(self.device).eval()

    def __call__(self, images: Sequence[bytes], prompt: str) -> str:
        return self.batch([(images, prompt)])[0]

    def batch(self, requests: Sequence[tuple[Sequence[bytes], str]]) -> list[str]:
        answers: list[str] = []
        for start in range(0, len(requests), self.batch_size):
            answers.extend(self._generate([[user_message(images, prompt)] for images, prompt in requests[start:start + self.batch_size]]))
        return answers

    def chat(self, messages: Sequence[Message]) -> str:
        return self._generate([messages])[0]

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Text as it is generated: generation runs on a thread and hands tokens to this generator."""
        import threading

        from transformers import TextIteratorStreamer

        inputs = self._inputs([messages])
        streamer = TextIteratorStreamer(self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor,
                                        skip_prompt=True, skip_special_tokens=True)
        failure: list[BaseException] = []

        def run() -> None:
            try:
                with self.torch.inference_mode():
                    self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, streamer=streamer)
            except BaseException as error:  # noqa: BLE001 - re-raised in the caller's thread
                failure.append(error)
                streamer.end()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        yield from (chunk for chunk in streamer if chunk)
        worker.join()
        if failure:
            raise RuntimeError(f"{self.model_id} failed: {type(failure[0]).__name__}: {failure[0]}") from failure[0]

    def _conversation(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        from PIL import Image

        converted = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                converted.append({"role": message["role"], "content": [{"type": "text", "text": content}]})
                continue
            parts: list[dict[str, Any]] = []
            for part in content:
                if part["type"] == "image":
                    with Image.open(io.BytesIO(part["data"])) as source:
                        parts.append({"type": "image", "image": source.convert("RGB")})
                else:
                    parts.append({"type": "text", "text": part["text"]})
            converted.append({"role": message["role"], "content": parts})
        return converted

    def _inputs(self, conversations: Sequence[Sequence[Message]]) -> Any:
        try:
            return self.processor.apply_chat_template(
                [self._conversation(messages) for messages in conversations], add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", processor_kwargs={"padding": True},
            ).to(self.device)
        except ValueError as error:
            raise RuntimeError(f"{self.model_id} could not read the request: {error}") from error

    def _generate(self, conversations: Sequence[Sequence[Message]]) -> list[str]:
        inputs = self._inputs(conversations)
        try:
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        except (self.torch.cuda.OutOfMemoryError, ValueError) as error:
            if self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()
            raise RuntimeError(f"{self.model_id} failed: {type(error).__name__}: {error}") from error
        generated = output[:, inputs["input_ids"].shape[1]:]
        return [text.strip() for text in self.processor.batch_decode(generated, skip_special_tokens=True)]
