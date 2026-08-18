"""Dataset adapters. All expose the same interface (see base.AnomalyDataset)."""
from .base import AnomalyDataset, FrameSequence

__all__ = ["AnomalyDataset", "FrameSequence", "get_dataset"]


def get_dataset(config) -> "AnomalyDataset":
    """Factory: build the dataset adapter named in the config."""
    name = config.dataset
    if name == "mvtec":
        from .mvtec import MVTecDataset
        return MVTecDataset(config.data_root, config.category)
    if name == "shanghaitech":
        from .shanghaitech import ShanghaiTechDataset
        return ShanghaiTechDataset(config.data_root, frame_step=config.frame_step)
    if name == "avenue":
        from .avenue import AvenueDataset
        return AvenueDataset(config.data_root, frame_step=config.frame_step)
    if name == "synthetic":
        from .base import SyntheticDataset
        return SyntheticDataset(seed=config.seed)
    raise ValueError(f"Unknown dataset: {name!r}")
