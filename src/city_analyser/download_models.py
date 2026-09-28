"""Pre-download segmentation weights into the local model cache.

Files land in the Hugging Face cache layout under ``models/``, which is the
``cache_dir`` used by building analysis, so later runs work offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .building_analysis import HIGH_ACCURACY_MODEL_ID


SEGFORMER_CITYSCAPES = {f"b{index}": f"nvidia/segformer-b{index}-finetuned-cityscapes-1024-1024" for index in range(6)}
MODEL_ALIASES = {**SEGFORMER_CITYSCAPES, "high": HIGH_ACCURACY_MODEL_ID}
DEFAULT_ALIASES = ("b0", "b1", "b2", "b5", "high")


def _weight_patterns(api: object, model_id: str, revision: str | None) -> list[str]:
    """Fetch configs, tokenizer files and one PyTorch weight format, preferring safetensors."""
    names = {sibling.rfilename for sibling in api.model_info(model_id, revision=revision).siblings}  # type: ignore[attr-defined]
    weights = "model.safetensors" if "model.safetensors" in names else "pytorch_model.bin"
    if weights not in names:
        raise RuntimeError(f"{model_id} publishes no PyTorch weights.")
    # Text-prompted models (Grounding DINO) also need their tokenizer vocabulary.
    return ["*.json", "*.txt", "*.model", weights]


def download(model_ids: list[str], model_root: Path, revision: str | None) -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:
        raise SystemExit("huggingface_hub is missing. Run uv sync, then retry.") from error
    api = HfApi()
    for model_id in model_ids:
        print(f"Downloading {model_id}...", flush=True)
        path = snapshot_download(
            model_id,
            revision=revision,
            cache_dir=model_root,
            allow_patterns=_weight_patterns(api, model_id, revision),
        )
        print(f"  -> {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download building-segmentation weights into the local cache.")
    parser.add_argument(
        "models", nargs="*", default=list(DEFAULT_ALIASES),
        help=f"aliases ({', '.join(MODEL_ALIASES)}), 'all', or Hugging Face model IDs (default: {' '.join(DEFAULT_ALIASES)})",
    )
    parser.add_argument("--model-root", type=Path, default=Path("models"), help="model cache directory (default: models)")
    parser.add_argument("--revision", default=None, help="optional immutable commit to download")
    args = parser.parse_args()
    requested = list(MODEL_ALIASES) if args.models == ["all"] else args.models
    download([MODEL_ALIASES.get(name, name) for name in requested], args.model_root, args.revision)


if __name__ == "__main__":
    main()
