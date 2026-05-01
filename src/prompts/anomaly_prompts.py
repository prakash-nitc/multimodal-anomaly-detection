"""
Anomaly Detection Prompt Templates
===================================
Text prompt templates for CLIP-based zero-shot anomaly detection.
Provides normal and abnormal prompt ensembles for each MVTec AD category.

Based on prompt strategies from:
- WinCLIP (Jeong et al., CVPR 2023): Compositional Prompt Ensembles
- AnomalyCLIP (Zhou et al., ICLR 2024): Object-agnostic prompting
"""

from typing import List, Dict


# ============================================================
# Human-readable display names for MVTec AD categories
# ============================================================
CATEGORY_DISPLAY_NAMES: Dict[str, str] = {
    "carpet": "carpet",
    "grid": "grid",
    "leather": "leather",
    "tile": "tile",
    "wood": "wood",
    "bottle": "bottle",
    "cable": "cable",
    "capsule": "capsule",
    "hazelnut": "hazelnut",
    "metal_nut": "metal nut",
    "pill": "pill",
    "screw": "screw",
    "toothbrush": "toothbrush",
    "transistor": "transistor",
    "zipper": "zipper",
}


# ============================================================
# Normal state descriptors (used across all categories)
# ============================================================
NORMAL_STATES = [
    "good",
    "perfect",
    "flawless",
    "pristine",
    "normal",
    "unblemished",
]

# ============================================================
# Abnormal state descriptors (used across all categories)
# ============================================================
ABNORMAL_STATES = [
    "damaged",
    "defective",
    "broken",
    "flawed",
    "abnormal",
    "imperfect",
]


# ============================================================
# Prompt templates — {state} and {object} are filled at runtime
# ============================================================
PROMPT_TEMPLATES = [
    "a photo of a {state} {object}",
    "a {state} {object}",
    "a photo of a {state} {object} for quality inspection",
    "a close-up photo of a {state} {object}",
]


def get_normal_prompts(category: str) -> List[str]:
    """Generate ensemble of normal-state text prompts for a category.

    Args:
        category: MVTec AD category name (e.g., 'bottle', 'metal_nut').

    Returns:
        List of text prompts describing the normal state.
    """
    obj_name = CATEGORY_DISPLAY_NAMES.get(category, category)
    prompts = []
    for template in PROMPT_TEMPLATES:
        for state in NORMAL_STATES:
            prompts.append(template.format(state=state, object=obj_name))
    return prompts


def get_abnormal_prompts(category: str) -> List[str]:
    """Generate ensemble of abnormal-state text prompts for a category.

    Args:
        category: MVTec AD category name (e.g., 'bottle', 'metal_nut').

    Returns:
        List of text prompts describing the anomalous state.
    """
    obj_name = CATEGORY_DISPLAY_NAMES.get(category, category)
    prompts = []
    for template in PROMPT_TEMPLATES:
        for state in ABNORMAL_STATES:
            prompts.append(template.format(state=state, object=obj_name))
    return prompts


def get_prompt_pair(category: str) -> tuple:
    """Get both normal and abnormal prompts for a category.

    Returns:
        (normal_prompts, abnormal_prompts) tuple of lists.
    """
    return get_normal_prompts(category), get_abnormal_prompts(category)


# Quick debug utility
if __name__ == "__main__":
    for cat in ["bottle", "metal_nut", "carpet"]:
        normal, abnormal = get_prompt_pair(cat)
        print(f"\n{'='*60}")
        print(f"Category: {cat}")
        print(f"Normal prompts ({len(normal)}):")
        for p in normal[:4]:
            print(f"  - {p}")
        print(f"  ... ({len(normal)} total)")
        print(f"Abnormal prompts ({len(abnormal)}):")
        for p in abnormal[:4]:
            print(f"  - {p}")
        print(f"  ... ({len(abnormal)} total)")
