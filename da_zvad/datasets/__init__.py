"""Dataset adapters. All expose the same interface (see base.AnomalyDataset)."""
from .base import AnomalyDataset, FrameSequence

__all__ = ["AnomalyDataset", "FrameSequence", "get_dataset"]


def get_dataset(config) -> "AnomalyDataset":
    """Factory: build the dataset adapter named in the config."""
    name = config.dataset
    if name == "mvtec":
        from .mvtec import MVTecDataset
        return MVTecDataset(config.data_root, config.category)
    if name in ("shanghaitech", "avenue"):
        from .video import VideoDataset
        return VideoDataset(config.data_root, sample_fps=config.sample_fps, name=name)
    if name == "synthetic":
        from .base import SyntheticDataset
        return SyntheticDataset(seed=config.seed)
    raise ValueError(f"Unknown dataset: {name!r}")
