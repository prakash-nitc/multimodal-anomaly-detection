"""M1 - Frozen CLIP visual encoder.

Wraps OpenCLIP. The image is scored by the softmax probability of matching the
"abnormal" text ensemble vs. the "normal" ensemble in CLIP's joint space -- the
language prior is the classifier, so no training is required.

Heavy dependencies (torch, open_clip) are imported lazily so the rest of the
framework (temporal, context, metrics, demo) runs without a GPU or model download.
"""
from __future__ import annotations

from typing import List, Sequence


class CLIPEncoder:
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self._device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    # -- lazy initialization --
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        import open_clip

        self.device = self._device if torch.cuda.is_available() else "cpu"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode_text_ensemble(self, prompts: Sequence[str]):
        """Encode and mean-pool a list of prompts into one unit embedding."""
        import torch
        import torch.nn.functional as F

        self._ensure_loaded()
        with torch.no_grad():
            tokens = self._tokenizer(list(prompts)).to(self.device)
            feats = F.normalize(self._model.encode_text(tokens), dim=-1)
            return F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)

    def score_frame(self, image, normal_embed, abnormal_embed) -> float:
        """Anomaly score for one frame (PIL image or file path) = P(abnormal)."""
        import torch
        import torch.nn.functional as F

        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image).convert("RGB")

        self._ensure_loaded()
        with torch.no_grad():
            x = self._preprocess(image).unsqueeze(0).to(self.device)
            img = F.normalize(self._model.encode_image(x), dim=-1)
            text = torch.cat([normal_embed, abnormal_embed], dim=0)
            sim = img @ text.T * self._model.logit_scale.exp()
            probs = sim.softmax(dim=-1)
            return float(probs[0, 1].item())

    def score_frames(self, images, normal_prompts: List[str], abnormal_prompts: List[str]):
        """Score a sequence of PIL images -> list of per-frame anomaly scores."""
        import numpy as np

        normal_embed = self.encode_text_ensemble(normal_prompts)
        abnormal_embed = self.encode_text_ensemble(abnormal_prompts)
        return np.array([self.score_frame(im, normal_embed, abnormal_embed) for im in images])

    def __repr__(self) -> str:
        return f"CLIPEncoder({self.model_name}, {self.pretrained})"
