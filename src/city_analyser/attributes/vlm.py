"""Vision-language provider: one structured question per crop, pooled per building.

The model is reached through ``generate(images, prompt) -> text``, so nothing
here imports PyTorch or runs untrusted model code. ``OpenAICompatibleClient``
talks to any OpenAI-style chat endpoint, e.g. a vLLM server hosting an
open InternVL3 or Qwen3-VL checkpoint on a cloud GPU. Answers are validated
against the taxonomy; anything else is an abstention for that view. A model's
answer is a vote, not a calibrated probability.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .fusion import DEFAULT_MIN_SHARE, fuse_categorical, fuse_numeric
from .models import BuildingAttributes
from .taxonomy import BUILDING_TYPES, CATEGORICAL_ATTRIBUTES, SURFACE_MATERIALS

PROMPT = (
    "Analyze the building shown in the image. Answer only with JSON using exactly these keys: "
    '{"building_type": one of %s, "alternate_building_type": second most likely type from the same list, '
    '"surface_material": one of %s, "alternate_surface_material": second most likely material from the same list, '
    '"floors": integer number of floors above ground, "construction_year": four-digit year, or null if you cannot tell}. '
    "Use null for any value that is not visible."
) % (json.dumps(list(BUILDING_TYPES)), json.dumps(list(SURFACE_MATERIALS)))
ALTERNATE_WEIGHT = 0.5
MAX_FLOORS = 150
GenerateFunction = Callable[[Sequence[bytes], str], str]


@dataclass(frozen=True)
class Crop:
    """One image of a building, with the weight its ranking gives it and a reference for the report."""

    image: bytes
    weight: float
    reference: dict[str, Any]


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _choice(value: Any, classes: Sequence[str]) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    return text if text in classes else None


def parse_answer(text: str) -> dict[str, Any]:
    """Validated fields of one model answer; invalid or missing ones are ``None``."""
    payload = _extract_json(text) or {}
    floors, year = payload.get("floors"), payload.get("construction_year")
    floors = float(floors) if isinstance(floors, (int, float)) and not isinstance(floors, bool) and 0 <= floors <= MAX_FLOORS else None
    year = float(year) if isinstance(year, (int, float)) and not isinstance(year, bool) and 1000 <= year <= 2100 else None
    return {
        "building_type": _choice(payload.get("building_type"), BUILDING_TYPES),
        "alternate_building_type": _choice(payload.get("alternate_building_type"), BUILDING_TYPES),
        "surface_material": _choice(payload.get("surface_material"), SURFACE_MATERIALS),
        "alternate_surface_material": _choice(payload.get("alternate_surface_material"), SURFACE_MATERIALS),
        "floors": floors,
        "construction_year": year,
    }


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


class VLMProvider:
    def __init__(self, generate: GenerateFunction, name: str, *, min_views: int = 1, min_share: float = DEFAULT_MIN_SHARE):
        self.generate, self.name, self.min_views, self.min_share = generate, name, min_views, min_share

    def predict(self, building_id: str, crops: Sequence[Crop]) -> BuildingAttributes:
        answers = [(parse_answer(self.generate([crop.image], PROMPT)), crop) for crop in crops]
        predictions: dict[str, Any] = {}
        for attribute, classes in CATEGORICAL_ATTRIBUTES.items():
            predictions[attribute] = fuse_categorical(
                attribute, classes, [(view_votes(answer, attribute), crop.weight) for answer, crop in answers], self.name,
                min_views=self.min_views, min_share=self.min_share,
            )
        for attribute in ("floors", "construction_year"):
            predictions[attribute] = fuse_numeric(attribute, [(answer[attribute], crop.weight) for answer, crop in answers], self.name, min_views=self.min_views)
        return BuildingAttributes(building_id, self.name, predictions, [crop.reference for crop in crops])


class OpenAICompatibleClient:
    """Chat-completions client for a self-hosted or cloud multimodal model server."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout_seconds: int = 120):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("the model server URL must be an HTTP(S) base URL without query parameters")
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model, self.api_key, self.timeout_seconds = model, api_key, timeout_seconds

    def __call__(self, images: Sequence[bytes], prompt: str) -> str:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")}} for image in images
        ]
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 300}
        headers = {"Content-Type": "application/json", "User-Agent": "local-panorama-building-viewer/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - administrator-configured endpoint
                payload = json.loads(response.read().decode("utf-8"))
            return str(payload["choices"][0]["message"]["content"])
        except HTTPError as error:
            raise RuntimeError(f"Model server failed ({error.code}): {error.reason}") from error
        except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Model server returned no usable answer: {error}") from error
