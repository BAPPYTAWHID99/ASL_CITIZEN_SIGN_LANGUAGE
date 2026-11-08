"""Application startup and globally shared ASL inference resources."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import torch
from django.apps import AppConfig

from train_bigru import BiGRUAttentionClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_bigru_attention_model.pt"
LABEL_MAP_PATH = PROJECT_ROOT / "models" / "label_map.json"
FALLBACK_LABEL_MAP_PATH = PROJECT_ROOT / "metadata" / "id_to_label.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_SHAPE = (200, 450)
INFERENCE_LOCK = threading.Lock()


def load_model() -> BiGRUAttentionClassifier:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"ASL model checkpoint not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid ASL model checkpoint: {MODEL_PATH}")
    model = BiGRUAttentionClassifier().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_label_map() -> dict[int, str]:
    path = LABEL_MAP_PATH if LABEL_MAP_PATH.is_file() else FALLBACK_LABEL_MAP_PATH
    if not path.is_file():
        raise FileNotFoundError(f"ASL label map not found: {LABEL_MAP_PATH}")
    with path.open(encoding="utf-8") as file:
        raw_mapping: Any = json.load(file)
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"ASL label map must be a JSON object: {path}")
    label_map = {int(index): str(word) for index, word in raw_mapping.items()}
    if len(label_map) != 200 or set(label_map) != set(range(200)):
        raise ValueError("ASL label map must contain exactly class IDs 0 through 199.")
    return label_map


class TranslatorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "translator"

    def ready(self) -> None:
        global MODEL, LABEL_MAP
        MODEL = load_model()
        LABEL_MAP = load_label_map()
