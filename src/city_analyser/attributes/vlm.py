"""Vision-language provider: one structured question per crop, pooled per building.

The model is reached through a ``ModelClient``: ``chat(messages) -> text``,
``stream(messages)`` yielding text as it is generated, and the older
``generate(images, prompt) -> text`` call. ``OpenAICompatibleClient`` talks to
any OpenAI-style chat endpoint (vLLM, Ollama, LM Studio, a cloud API);
``local_vlm`` runs open checkpoints in process with Transformers. Nothing here
imports PyTorch or runs model code downloaded with weights.

The prompt gives the model what an engineer knows about the place (see
``regional``) and asks it to reason in plain prose before it commits to a JSON
answer, as field notes would. That prose is streamed to the viewer while the
model works and stored with each view; it explains an answer, it is not
evidence by itself. Answers are validated against the taxonomy; anything else
is an abstention for that view. A model's answer is a vote, not a calibrated
probability.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .fusion import DEFAULT_MIN_SHARE, fuse_categorical, fuse_numeric
from .models import BuildingAttributes
from .regional import GUIDANCE_VERSION, guidance_for
from .taxonomy import BUILDING_TYPES, CATEGORICAL_ATTRIBUTES, GEM_LLRS, GEM_MATERIALS, SURFACE_MATERIALS, TAXONOMY_VERSION

# Bump when the prompt or the parsing changes: it is part of every result's input fingerprint.
PROMPT_VERSION = 4


def _codes(codes: dict[str, str]) -> str:
    return "; ".join(f"{code} = {meaning}" for code, meaning in codes.items())


ROLE = (
    "You are a structural engineer surveying buildings from street-level photos for a seismic exposure model "
    "that uses the GEM Building Taxonomy v3. The image shows one building; other buildings may be painted gray."
)
TASK = (
    "First think it through in plain prose, as field notes: three to six short sentences on the storeys you count, "
    "the cues for the structure, the likely period, and anything that limits what you can tell. "
    "Then, on its own line, give one JSON object with exactly these keys: "
    '{"observations": one sentence summarising the cues you used, '
    '"floors": integer number of storeys above ground, '
    '"gem_material": GEM code of the material of the lateral load-resisting system, '
    '"alternate_gem_material": second most likely code, '
    '"gem_llrs": GEM code of the type of lateral load-resisting system, '
    '"alternate_gem_llrs": second most likely code, '
    '"building_type": one of %s, "alternate_building_type": second most likely from the same list, '
    '"surface_material": one of %s (what covers the facade), "alternate_surface_material": second most likely from the same list, '
    '"construction_year": four-digit year}. '
    "GEM material codes: %s. GEM lateral load-resisting system codes: %s. "
    "Use null for any value you cannot judge from the image; do not guess a structure the facade hides. "
    "Write nothing after the JSON."
) % (json.dumps(list(BUILDING_TYPES)), json.dumps(list(SURFACE_MATERIALS)), _codes(GEM_MATERIALS), _codes(GEM_LLRS))
# The question without a region or a capture context.
PROMPT = f"{ROLE} {guidance_for(None)} {TASK}"
ALTERNATE_WEIGHT = 0.5
MAX_FLOORS = 150
MAX_OBSERVATION_CHARS = 600
MAX_REASONING_CHARS = 2000
# Field notes plus the JSON answer.
MAX_ANSWER_TOKENS = 900

# A chat message: {"role": "system" | "user" | "assistant", "content": text or parts};
# a part is {"type": "text", "text": ...} or {"type": "image", "data": jpeg bytes}.
Message = dict[str, Any]
GenerateFunction = Callable[[Sequence[bytes], str], str]


class ModelClient(Protocol):
    def __call__(self, images: Sequence[bytes], prompt: str) -> str: ...

    def chat(self, messages: Sequence[Message]) -> str: ...

    def stream(self, messages: Sequence[Message]) -> Iterator[str]: ...


def user_message(images: Sequence[bytes], text: str) -> Message:
    return {"role": "user", "content": [*({"type": "image", "data": image} for image in images), {"type": "text", "text": text}]}


def build_prompt(context: str | None = None, region: str | None = None) -> str:
    """The question for one view: role, the region's construction history, the task and the capture context.

    ``context`` (where and when the photo was taken) must never carry OSM tags: the evaluation scores answers against them.
    """
    prompt = f"{ROLE} {guidance_for(region)} {TASK}"
    return f"{prompt} Context: {context}." if context else prompt


@dataclass(frozen=True)
class Crop:
    """One image of a building, with the weight its ranking gives it, a reference for the report and the prompt context."""

    image: bytes
    weight: float
    reference: dict[str, Any]
    context: str | None = None
    region: str | None = None


def _json_spans(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))


def _extract_json(text: str) -> dict[str, Any] | None:
    # The answer is the last object, after any reasoning.
    for match in reversed(_json_spans(text)):
        try:
            # strict=False: models write raw line breaks inside the observations string.
            payload = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def visible_reasoning(text: str) -> str:
    """The field notes before the JSON answer (also on a partial, still-streaming answer)."""
    cut = len(text)
    for marker in ("```", "\n{", "{\""):
        index = text.find(marker)
        if index >= 0:
            cut = min(cut, index)
    if text.lstrip().startswith("{"):
        cut = 0
    notes = re.sub(r"<think>.*?(</think>|$)", "", text[:cut], flags=re.DOTALL)
    return notes.strip()


def _choice(value: Any, classes: Sequence[str]) -> str | None:
    """The class ``value`` names, ignoring case, spaces and hyphens (GEM codes are upper case, the others lower case)."""
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_") if value is not None else ""
    return next((name for name in classes if name.lower() == text), None)


def _number(value: Any) -> float | None:
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def parse_answer(text: str) -> dict[str, Any]:
    """Validated fields of one model answer; invalid or missing ones are ``None``."""
    payload = _extract_json(text) or {}
    floors, year = _number(payload.get("floors")), _number(payload.get("construction_year"))
    floors = floors if floors is not None and 0 < floors <= MAX_FLOORS else None
    year = year if year is not None and 1000 <= year <= 2100 else None
    observations = payload.get("observations")
    answer: dict[str, Any] = {
        "observations": " ".join(str(observations).split())[:MAX_OBSERVATION_CHARS] if isinstance(observations, str) and observations.strip() else None,
        "floors": floors,
        "construction_year": year,
    }
    for attribute, classes in CATEGORICAL_ATTRIBUTES.items():
        answer[attribute] = _choice(payload.get(attribute), classes)
        answer[f"alternate_{attribute}"] = _choice(payload.get(f"alternate_{attribute}"), classes)
    return answer


def view_votes(answer: dict[str, Any], attribute: str) -> dict[str, float]:
    """Normalized votes of one answer: 1 for its first choice, ``ALTERNATE_WEIGHT`` for its second."""
    votes: dict[str, float] = {}
    first, second = answer.get(attribute), answer.get(f"alternate_{attribute}")
    if first:
        votes[first] = 1.0
    if second and second != first:
        votes[second] = ALTERNATE_WEIGHT
    total = sum(votes.values())
    return {name: value / total for name, value in votes.items()} if total else {}


def inputs_fingerprint(provider: str, crops: Sequence[Crop]) -> str:
    """What a result depends on: provider, prompt, regional guidance, taxonomy and every crop with its weight and context.

    A building is asked again only when this changes, e.g. after a new
    panorama gives it a better view.
    """
    payload = {
        "provider": provider, "prompt_version": PROMPT_VERSION, "guidance_version": GUIDANCE_VERSION, "taxonomy_version": TAXONOMY_VERSION,
        "crops": [[hashlib.sha256(crop.image).hexdigest(), round(crop.weight, 6), crop.context, crop.region] for crop in crops],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# Live analysis events: {"type": "view", "index", "reference"}, {"type": "text", "index", "text"} (a chunk),
# {"type": "answer", "index", "answer", "reasoning", "error"}.
EventSink = Callable[[dict[str, Any]], None]


class VLMProvider:
    """Asks the model about every crop, pools the answers and abstains where they are weak.

    A view whose request fails after ``attempts`` tries is recorded with its
    error and counts as no answer; it never stops the other views. When the
    client has ``batch(requests) -> answers``, views go in one batch; with an
    ``on_event`` sink and a client that can ``stream``, each view's text is
    reported as it is generated.
    """

    def __init__(
        self, generate: GenerateFunction, name: str, *, min_views: int = 1, min_share: float = DEFAULT_MIN_SHARE,
        attempts: int = 2, retry_pause_s: float = 2.0,
    ):
        self.generate, self.name, self.min_views, self.min_share = generate, name, min_views, min_share
        self.attempts, self.retry_pause_s = max(1, attempts), retry_pause_s

    def _retrying(self, call: Callable[[], str]) -> tuple[str | None, str | None]:
        for attempt in range(1, self.attempts + 1):
            try:
                return call(), None
            except RuntimeError as error:
                if attempt == self.attempts:
                    return None, str(error)
                time.sleep(self.retry_pause_s)
        return None, "no attempt made"

    def _ask_all(self, crops: Sequence[Crop]) -> list[tuple[str | None, str | None]]:
        requests = [([crop.image], build_prompt(crop.context, crop.region)) for crop in crops]
        batch = getattr(self.generate, "batch", None)
        if batch is not None and len(requests) > 1:
            try:
                return [(text, None) for text in batch(requests)]
            except RuntimeError:
                pass  # asked one by one below, so one bad view cannot fail the others
        return [self._retrying(lambda images=images, prompt=prompt: self.generate(images, prompt)) for images, prompt in requests]

    def _stream_all(self, crops: Sequence[Crop], on_event: EventSink) -> list[tuple[str | None, str | None]]:
        stream = getattr(self.generate, "stream", None)
        results = []
        for index, crop in enumerate(crops):
            on_event({"type": "view", "index": index, "reference": crop.reference})
            message = user_message([crop.image], build_prompt(crop.context, crop.region))

            def call() -> str:
                if stream is None:
                    text = self.generate([crop.image], build_prompt(crop.context, crop.region))
                    on_event({"type": "text", "index": index, "text": text})
                    return text
                chunks = []
                for chunk in stream([message]):
                    chunks.append(chunk)
                    on_event({"type": "text", "index": index, "text": chunk})
                return "".join(chunks)

            text, error = self._retrying(call)
            answer = parse_answer(text or "")
            on_event({"type": "answer", "index": index, "answer": {k: v for k, v in answer.items() if v is not None},
                      "reasoning": visible_reasoning(text or ""), "error": error})
            results.append((text, error))
        return results

    def predict(self, building_id: str, crops: Sequence[Crop], on_event: EventSink | None = None) -> BuildingAttributes:
        texts = self._stream_all(crops, on_event) if on_event is not None else self._ask_all(crops)
        answers: list[tuple[dict[str, Any], Crop]] = []
        evidence: list[dict[str, Any]] = []
        for (text, error), crop in zip(texts, crops):
            answer = parse_answer(text) if text is not None else parse_answer("")
            answers.append((answer, crop))
            view = {**crop.reference, "answer": {key: value for key, value in answer.items() if value is not None}}
            reasoning = visible_reasoning(text or "")
            if reasoning:
                view["reasoning"] = reasoning[:MAX_REASONING_CHARS]
            if error is not None:
                view["error"] = error[:300]
            elif not any(value is not None for key, value in answer.items() if key != "observations"):
                view["error"] = "the answer had no valid field"
            evidence.append(view)
        predictions: dict[str, Any] = {}
        for attribute, classes in CATEGORICAL_ATTRIBUTES.items():
            predictions[attribute] = fuse_categorical(
                attribute, classes, [(view_votes(answer, attribute), crop.weight) for answer, crop in answers], self.name,
                min_views=self.min_views, min_share=self.min_share,
            )
        for attribute in ("floors", "construction_year"):
            predictions[attribute] = fuse_numeric(attribute, [(answer[attribute], crop.weight) for answer, crop in answers], self.name, min_views=self.min_views)
        run = {"prompt_version": PROMPT_VERSION, "guidance_version": GUIDANCE_VERSION, "inputs_sha256": inputs_fingerprint(self.name, crops)}
        return BuildingAttributes(building_id, self.name, predictions, evidence, run=run)


class OpenAICompatibleClient:
    """Chat-completions client for a self-hosted or cloud multimodal model server.

    ``options`` are extra request fields the server understands, e.g.
    ``{"reasoning_effort": "none"}`` for Ollama models that would otherwise
    spend the answer budget thinking out of sight.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout_seconds: int = 300, options: dict[str, Any] | None = None):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("the model server URL must be an HTTP(S) base URL without query parameters")
        base = base_url.rstrip("/")
        # Accept both https://host and https://host/v1 (how most providers document their base URL).
        self.url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        self.model, self.api_key, self.timeout_seconds, self.options = model, api_key, timeout_seconds, dict(options or {})

    @staticmethod
    def _content(content: Any) -> Any:
        if isinstance(content, str):
            return content
        parts = []
        for part in content:
            if part["type"] == "image":
                parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(part["data"]).decode("ascii")}})
            else:
                parts.append({"type": "text", "text": part["text"]})
        return parts

    def _request(self, messages: Sequence[Message], stream: bool) -> Request:
        body = {
            "model": self.model, "messages": [{"role": message["role"], "content": self._content(message["content"])} for message in messages],
            "temperature": 0, "max_tokens": MAX_ANSWER_TOKENS, "stream": stream, **self.options,
        }
        headers = {"Content-Type": "application/json", "User-Agent": "local-panorama-building-viewer/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")

    def chat(self, messages: Sequence[Message]) -> str:
        try:
            with urlopen(self._request(messages, False), timeout=self.timeout_seconds) as response:  # noqa: S310 - administrator-configured endpoint
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"].get("content")
        except HTTPError as error:
            raise RuntimeError(f"Model server failed ({error.code}): {error.reason}") from error
        except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as error:
            raise RuntimeError(f"Model server returned no usable answer: {error}") from error
        if not content:
            raise RuntimeError("Model server returned an empty answer (a reasoning model may have spent its budget thinking: set reasoning_effort to none)")
        return str(content)

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Text chunks as the server generates them (server-sent events)."""
        produced = False
        try:
            with urlopen(self._request(messages, True), timeout=self.timeout_seconds) as response:  # noqa: S310 - administrator-configured endpoint
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or [{}]
                    chunk = (choices[0].get("delta") or {}).get("content")
                    if chunk:
                        produced = True
                        yield chunk
        except HTTPError as error:
            raise RuntimeError(f"Model server failed ({error.code}): {error.reason}") from error
        except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError, TypeError, AttributeError) as error:
            raise RuntimeError(f"Model server stream failed: {error}") from error
        if not produced:
            raise RuntimeError("Model server returned an empty answer (a reasoning model may have spent its budget thinking: set reasoning_effort to none)")

    def __call__(self, images: Sequence[bytes], prompt: str) -> str:
        return self.chat([user_message(images, prompt)])
