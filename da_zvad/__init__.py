"""DA-ZVAD — Domain-Adaptive Zero-shot Video Anomaly Detection.

A modular, configuration-driven framework for training-free anomaly detection
with vision-language and large language models.

Pipeline:  frames -> [M1 CLIP encoder] -> [M2 temporal] -> [M3 context] -> [M4 LLM reasoning]
           -> per-frame anomaly score + natural-language explanation.

Each module can be toggled on/off via ``DAZVADConfig`` so the same code path
produces the full ablation study by configuration alone.
"""

from .config import DAZVADConfig

__version__ = "0.1.0"
__all__ = ["DAZVADConfig", "__version__"]
