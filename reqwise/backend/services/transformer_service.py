from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from reqwise.backend.settings import ROBERTA_AMBIGUITY_DIR, ROBERTA_SUBTYPE_DIR


@dataclass
class TransformerPrediction:
    label: str
    confidence: float


class _CheckpointClassifier:
    def __init__(self, checkpoints_dir: Path, task_name: str) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.task_name = task_name
        self.loaded = False
        self.last_error: Optional[str] = None

        self.tokenizer = None
        self.model = None
        self.torch = None
        self.id2label: dict[int, str] = {}

    @staticmethod
    def _checkpoint_num(path: Path) -> int:
        match = re.search(r"checkpoint-(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    def _pick_checkpoint(self) -> Optional[Path]:
        if not self.checkpoints_dir.exists():
            return None
        candidates = [p for p in self.checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        if not candidates:
            return None
        return sorted(candidates, key=self._checkpoint_num, reverse=True)[0]

    def load(self) -> bool:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            self.loaded = False
            self.last_error = f"transformers/torch unavailable: {exc}"
            return False

        ckpt = self._pick_checkpoint()
        if ckpt is None:
            self.loaded = False
            self.last_error = f"No checkpoint directory found in {self.checkpoints_dir}"
            return False

        try:
            # Prefer local cache to avoid network dependency in inference.
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", local_files_only=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

            model = AutoModelForSequenceClassification.from_pretrained(str(ckpt))
            model.eval()

            raw_id2label = getattr(model.config, "id2label", {}) or {}
            parsed_id2label: dict[int, str] = {}
            for k, v in raw_id2label.items():
                try:
                    parsed_id2label[int(k)] = str(v)
                except Exception:
                    continue

            self.tokenizer = tokenizer
            self.model = model
            self.torch = torch
            self.id2label = parsed_id2label
            self.loaded = True
            self.last_error = None
            return True
        except Exception as exc:
            self.loaded = False
            self.last_error = str(exc)
            return False

    def predict(self, text: str, max_length: int = 128) -> Optional[TransformerPrediction]:
        if not self.loaded or self.model is None or self.tokenizer is None or self.torch is None:
            return None

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        with self.torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probs = self.torch.softmax(logits, dim=-1).cpu().numpy()

        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        label = self.id2label.get(pred_id, str(pred_id))
        return TransformerPrediction(label=label, confidence=confidence)


class TransformerService:
    def __init__(self) -> None:
        self.subtype = _CheckpointClassifier(ROBERTA_SUBTYPE_DIR, "subtype")
        self.ambiguity = _CheckpointClassifier(ROBERTA_AMBIGUITY_DIR, "ambiguity")

    def load_models(self) -> None:
        self.subtype.load()
        self.ambiguity.load()

    @property
    def subtype_loaded(self) -> bool:
        return self.subtype.loaded

    @property
    def ambiguity_loaded(self) -> bool:
        return self.ambiguity.loaded

    def predict_subtype(self, text: str) -> Optional[TransformerPrediction]:
        return self.subtype.predict(text)

    def predict_ambiguity(self, text: str) -> Optional[TransformerPrediction]:
        return self.ambiguity.predict(text)
