"""M4 - LLaVA-based reasoner (the real implementation).

For each selected anomalous frame, a multimodal LLM (LLaVA 1.5) is shown the
frame plus the verbalized domain context and asked to describe, in one or two
sentences, what is unusual. The prompt explicitly allows "nothing seems wrong"
-- an anti-hallucination measure worth mentioning in the thesis.

Heavy dependencies (torch, transformers, bitsandbytes) are imported lazily and
the model loads in 4-bit on CUDA (~5.5 GB VRAM, fits a Kaggle T4). This class
is not meant to run on CPU-only machines; use StubReasoner there.
"""
from __future__ import annotations

from typing import Any

from .llm_reasoner import LLMReasoner


class LlavaReasoner(LLMReasoner):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 60,
    ):
        self.model_id = model_id
        self._device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    # -- lazy initialization --
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = self._device if torch.cuda.is_available() else "cpu"
        kwargs = {"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        if self.load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            kwargs["device_map"] = None
        self._model = LlavaForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        if "quantization_config" not in kwargs:
            self._model = self._model.to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def build_prompt(self, context: str, score: float) -> str:
        ctx = (context or "an unspecified scene").strip().rstrip(".")
        return (
            "USER: <image>\n"
            f"This frame comes from {ctx}. An automated detector flagged it as "
            f"anomalous (score {score:.2f}). In one or two sentences, describe the "
            "unusual or dangerous thing happening in the frame. If nothing seems "
            "wrong, say so.\nASSISTANT:"
        )

    def _generate(self, image, prompt: str) -> str:
        import torch

        self._ensure_loaded()
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
            self._model.device
        )
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        text = self._processor.decode(out[0], skip_special_tokens=True)
        # keep only the assistant's reply
        return text.split("ASSISTANT:")[-1].strip()

    def explain(self, frame: Any, context: str, score: float) -> str:
        if isinstance(frame, str):
            from PIL import Image
            frame = Image.open(frame).convert("RGB")
        if frame is None:
            return "[no frame available for explanation]"
        return self._generate(frame, self.build_prompt(context, score))

    def __repr__(self) -> str:
        return f"LlavaReasoner({self.model_id}, 4bit={self.load_in_4bit})"
