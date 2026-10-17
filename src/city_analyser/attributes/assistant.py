"""The viewer's assistant: analyze one building now, or talk about it.

Both use the same model as the survey (``ModelSettings``) and the same crops.
``analyze`` runs the vision-language provider on the building's views while
reporting each view's field notes as they are generated, then saves the answer
like a survey would. ``chat`` answers an engineer's questions with the
building's photos, its current record and the region's construction history in
context; it never changes the record, which only a reviewer's decision does.
One model call runs at a time: a local model has one GPU, a server one queue.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..building_inspection.crops import export_building_crops
from .card import building_card
from .cli import ModelSettings, load_crops, write_attributes
from .regional import guidance_for
from .vlm import ROLE, Message, VLMProvider, user_message

MAX_HISTORY = 16
MAX_MESSAGE_CHARS = 2000
CHAT_IMAGES = 3
CHAT_RULES = (
    "You are talking with an engineer who is checking this building's record. Answer in the engineer's language, "
    "briefly and concretely, and say which photo and which cue supports what you say. When the photos cannot settle "
    "a question, say so and say what would (another view, the cadastre, an inspection). You cannot change the record: "
    "if the engineer wants to correct it, tell them to use Review on the building card, and say whether the photos "
    "support the correction."
)


class AssistantUnavailable(RuntimeError):
    """No model is configured, or the building has nothing to show the model."""


def record_summary(card: dict[str, Any] | None) -> dict[str, Any]:
    """What the model is told about the current record: values, where they come from, and any review."""
    if not card:
        return {"record": "this building has not been described yet"}
    attributes = {
        name: {"value": None if entry.get("abstained") else entry.get("value"), "source": entry.get("source"),
               **({"why_undetermined": entry.get("reason")} if entry.get("abstained") else {})}
        for name, entry in card.get("attributes", {}).items()
    }
    summary: dict[str, Any] = {"building": card.get("building"), "gem": card.get("gem", {}).get("string"), "attributes": attributes}
    if card.get("review"):
        summary["reviewed_by_engineer"] = card["review"]
    notes = [view.get("reasoning") or view.get("answer", {}).get("observations") for view in card.get("views", [])]
    summary["model_notes_per_view"] = [note for note in notes if note]
    return summary


def chat_messages(card: dict[str, Any] | None, images: Sequence[bytes], history: Sequence[dict[str, Any]], region: str | None) -> list[Message]:
    """System context, the photos, then the conversation (validated and trimmed to the last ``MAX_HISTORY`` turns)."""
    turns = [
        {"role": item["role"], "content": str(item["content"])[:MAX_MESSAGE_CHARS]}
        for item in history if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and str(item.get("content", "")).strip()
    ][-MAX_HISTORY:]
    if not turns or turns[-1]["role"] != "user":
        raise ValueError("the conversation must end with a message from the engineer")
    system = f"{ROLE} {guidance_for(region)} {CHAT_RULES} Current record: {json.dumps(record_summary(card), ensure_ascii=False)}"
    photos = user_message(images[:CHAT_IMAGES], f"These are the best views of the building, best first ({min(len(images), CHAT_IMAGES)} photos).")
    # Some chat templates need alternating roles: the photos get a short acknowledgement.
    return [{"role": "system", "content": system}, photos, {"role": "assistant", "content": "I have the photos and the record."}, *turns]


class Assistant:
    def __init__(self, settings: ModelSettings):
        self.settings = settings
        self._client: Any = None
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {"available": True, "model": self.settings.model, "backend": self.settings.backend, "provider": self.settings.provider_name}

    def _model(self) -> Any:
        if self._client is None:
            self._client = self.settings.client()
        return self._client

    def _crops(self, data_root: Path, dataset_id: str, building_id: str, images_by_id: dict[str, dict[str, Any]]) -> list[Any]:
        dataset_dir = (data_root / dataset_id).resolve()
        crops = load_crops(dataset_dir, building_id, images_by_id)
        if not crops:
            export_building_crops(data_root, dataset_id, [building_id], mask_others=True)
            crops = load_crops(dataset_dir, building_id, images_by_id)
        if not crops:
            raise AssistantUnavailable(f"{building_id} has no analyzed view in this dataset to show the model")
        return crops

    def analyze(
        self, data_root: Path, dataset_id: str, building_id: str, images_by_id: dict[str, dict[str, Any]], on_event: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Run the model on every view now, report progress through ``on_event``, save and return the updated card."""
        crops = self._crops(data_root, dataset_id, building_id, images_by_id)
        on_event({"type": "start", "views": len(crops), "model": self.settings.model})
        with self._lock:
            result = VLMProvider(self._model(), self.settings.provider_name).predict(building_id, crops, on_event=on_event)
        dataset_dir = (data_root / dataset_id).resolve()
        write_attributes(dataset_dir, result)
        card = building_card(dataset_dir, dataset_id, building_id)
        if card is None:
            raise AssistantUnavailable(f"{building_id}: the answer could not be saved")
        return card

    def chat(
        self, data_root: Path, dataset_id: str, building_id: str, history: Sequence[dict[str, Any]],
        images_by_id: dict[str, dict[str, Any]], on_text: Callable[[str], None],
    ) -> str:
        """Answer the last message of ``history``, reporting the text as it is generated; returns the whole answer."""
        dataset_dir = (data_root / dataset_id).resolve()
        crops = self._crops(data_root, dataset_id, building_id, images_by_id)
        messages = chat_messages(building_card(dataset_dir, dataset_id, building_id), [crop.image for crop in crops], history, crops[0].region)
        chunks: list[str] = []
        with self._lock:
            model = self._model()
            stream = getattr(model, "stream", None)
            if stream is None:
                chunks.append(model.chat(messages))
                on_text(chunks[-1])
            else:
                for chunk in stream(messages):
                    chunks.append(chunk)
                    on_text(chunk)
        return "".join(chunks)
