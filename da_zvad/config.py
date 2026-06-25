"""Central configuration for a DA-ZVAD run.

One config object drives every experiment. Toggling ``use_temporal`` /
``use_context`` / ``use_reasoning`` is exactly how the ablation grid is produced
without changing any code.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class DAZVADConfig:
    # --- data ---
    dataset: str = "synthetic"              # synthetic | mvtec | shanghaitech | avenue
    data_root: Optional[str] = None
    category: Optional[str] = None          # e.g. MVTec category ("bottle")
    sample_fps: int = 2                     # video frame sampling rate

    # --- M1: visual encoder ---
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "laion2b_s32b_b82k"
    device: str = "cuda"                    # falls back to cpu automatically

    # --- domain / prompts ---
    domain: str = "generic"                 # generic | industrial | surveillance
    domain_description: str = "a generic scene"   # M3 verbalized context

    # --- module toggles (ablation switches) ---
    use_temporal: bool = True               # M2 temporal aggregation
    use_context: bool = True                # M3 verbalized domain context
    use_reasoning: bool = False             # M4 LLM reasoning (heavy; stub for now)

    # --- M2 temporal ---
    temporal_window: int = 5                # moving-average window (frames)

    # --- detection ---
    threshold: float = 0.5

    # --- misc ---
    seed: int = 0

    def tag(self) -> str:
        """Short identifier used to name result files for this configuration."""
        parts = [self.dataset]
        if self.category:
            parts.append(self.category)
        parts.append("M1")
        if self.use_temporal:
            parts.append(f"M2w{self.temporal_window}")
        if self.use_context:
            parts.append("M3")
        if self.use_reasoning:
            parts.append("M4")
        return "_".join(parts)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DAZVADConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
