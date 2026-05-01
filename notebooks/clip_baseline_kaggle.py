# ============================================================
# CLIP Zero-Shot Anomaly Detection Baseline — MVTec AD
# ============================================================
# Run this as a Kaggle Notebook with GPU enabled.
#
# Setup steps in Kaggle:
# 1. Create a new Notebook
# 2. Add dataset: kaggle.com/datasets/ipythonx/mvtec-ad
# 3. Enable GPU: Settings → Accelerator → GPU T4 x2
# 4. Paste this entire script into a cell and run
# ============================================================

# --- Cell 1: Install dependencies ---
# !pip install open-clip-torch -q

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 1. DATASET
# ============================================================

MVTEC_CATEGORIES = [
    "carpet", "grid", "leather", "tile", "wood",
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]

class MVTecDataset(Dataset):
    """MVTec AD test set loader."""

    def __init__(self, root_dir: str, category: str, split: str = "test", transform: Optional[Callable] = None):
        assert category in MVTEC_CATEGORIES, f"Unknown category '{category}'"
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()

    def _load_samples(self):
        split_dir = self.root_dir / self.category / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Not found: {split_dir}")
        for defect_type in sorted(os.listdir(split_dir)):
            defect_dir = split_dir / defect_type
            if not defect_dir.is_dir():
                continue
            label = 0 if defect_type == "good" else 1
            for img_file in sorted(os.listdir(defect_dir)):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((str(defect_dir / img_file), label))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, self.category, img_path

    @property
    def num_normal(self): return sum(1 for _, l in self.samples if l == 0)
    @property
    def num_anomalous(self): return sum(1 for _, l in self.samples if l == 1)


# ============================================================
# 2. PROMPT TEMPLATES
# ============================================================

CATEGORY_DISPLAY_NAMES = {
    "carpet": "carpet", "grid": "grid", "leather": "leather", "tile": "tile", "wood": "wood",
    "bottle": "bottle", "cable": "cable", "capsule": "capsule", "hazelnut": "hazelnut",
    "metal_nut": "metal nut", "pill": "pill", "screw": "screw", "toothbrush": "toothbrush",
    "transistor": "transistor", "zipper": "zipper",
}

NORMAL_STATES = ["good", "perfect", "flawless", "pristine", "normal", "unblemished"]
ABNORMAL_STATES = ["damaged", "defective", "broken", "flawed", "abnormal", "imperfect"]

PROMPT_TEMPLATES = [
    "a photo of a {state} {object}",
    "a {state} {object}",
    "a photo of a {state} {object} for quality inspection",
    "a close-up photo of a {state} {object}",
]

def get_prompts(category: str, states: list) -> List[str]:
    obj = CATEGORY_DISPLAY_NAMES.get(category, category)
    return [t.format(state=s, object=obj) for t in PROMPT_TEMPLATES for s in states]

def get_prompt_pair(category: str):
    return get_prompts(category, NORMAL_STATES), get_prompts(category, ABNORMAL_STATES)


# ============================================================
# 3. CLIP ZERO-SHOT DETECTOR
# ============================================================

class CLIPZeroShotDetector:
    def __init__(self, model_name="ViT-L-14", pretrained="laion2b_s32b_b82k", device=None):
        import open_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading OpenCLIP {model_name} ({pretrained}) on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("Model loaded!")

    @torch.no_grad()
    def encode_text_prompts(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
        mean_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
        return mean_feat

    @torch.no_grad()
    def compute_anomaly_scores(self, dataloader, normal_prompts, abnormal_prompts):
        normal_embed = self.encode_text_prompts(normal_prompts)
        abnormal_embed = self.encode_text_prompts(abnormal_prompts)
        text_embeds = torch.cat([normal_embed, abnormal_embed], dim=0)

        all_scores, all_labels = [], []
        for images, labels, _, _ in tqdm(dataloader, desc="Scoring", leave=False):
            images = images.to(self.device)
            img_feats = self.model.encode_image(images)
            img_feats = F.normalize(img_feats, dim=-1)
            sim = img_feats @ text_embeds.T * self.model.logit_scale.exp()
            probs = sim.softmax(dim=-1)
            all_scores.append(probs[:, 1].cpu().numpy())
            all_labels.append(labels.numpy())

        return np.concatenate(all_scores), np.concatenate(all_labels)


# ============================================================
# 4. METRICS
# ============================================================

def compute_auroc(scores, labels):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)

def compute_optimal_f1(scores, labels):
    best = {"f1": 0, "precision": 0, "recall": 0, "threshold": 0}
    for t in np.unique(scores):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
                "threshold": float(t),
            }
    return best


# ============================================================
# 5. MAIN — RUN EVALUATION
# ============================================================

# *** UPDATE THIS PATH ***
# On Kaggle, the dataset is typically mounted at:
DATA_ROOT = "/kaggle/input/mvtec-ad"

# If your Kaggle dataset has an extra folder level, try:
# DATA_ROOT = "/kaggle/input/mvtec-ad/mvtec_anomaly_detection"

# Check if the path exists and find the right root
if not os.path.exists(DATA_ROOT):
    # Try common Kaggle paths
    candidates = [
        "/kaggle/input/mvtec-ad",
        "/kaggle/input/mvtec-ad/mvtec_anomaly_detection",
        "/kaggle/input/mvtecad",
        "/kaggle/input/mvtec-anomaly-detection",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "bottle")):
            DATA_ROOT = c
            break

print(f"Data root: {DATA_ROOT}")
print(f"Contents: {os.listdir(DATA_ROOT)[:5]}...")

# Load model
detector = CLIPZeroShotDetector(model_name="ViT-L-14", pretrained="laion2b_s32b_b82k")

# Run all categories
results = {}
total_start = time.time()

for i, category in enumerate(MVTEC_CATEGORIES, 1):
    cat_dir = os.path.join(DATA_ROOT, category)
    if not os.path.exists(cat_dir):
        print(f"[{i}/15] SKIP: {category} (not found)")
        continue

    print(f"\n[{i}/15] {category}")

    dataset = MVTecDataset(DATA_ROOT, category, split="test", transform=detector.preprocess)
    print(f"  Samples: {len(dataset)} (normal={dataset.num_normal}, anomalous={dataset.num_anomalous})")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    normal_prompts, abnormal_prompts = get_prompt_pair(category)

    t0 = time.time()
    scores, labels = detector.compute_anomaly_scores(dataloader, normal_prompts, abnormal_prompts)
    elapsed = time.time() - t0

    auroc = compute_auroc(scores, labels)
    f1_res = compute_optimal_f1(scores, labels)

    results[category] = {
        "auroc": auroc, "f1": f1_res["f1"],
        "precision": f1_res["precision"], "recall": f1_res["recall"],
        "n_normal": dataset.num_normal, "n_anomalous": dataset.num_anomalous,
    }
    print(f"  AUROC: {auroc:.1%}  |  F1: {f1_res['f1']:.1%}  |  Time: {elapsed:.1f}s")

total_time = time.time() - total_start

# Mean
valid = {k: v for k, v in results.items() if not np.isnan(v["auroc"])}
mean_auroc = np.mean([v["auroc"] for v in valid.values()])
mean_f1 = np.mean([v["f1"] for v in valid.values()])
print(f"\n{'='*55}")
print(f"  MEAN AUROC: {mean_auroc:.1%}  |  MEAN F1: {mean_f1:.1%}")
print(f"  Total time: {total_time:.1f}s")
print(f"{'='*55}")


# ============================================================
# 6. RESULTS TABLE & PLOT
# ============================================================

# DataFrame
rows = [{"Category": k, "AUROC": v["auroc"], "F1": v["f1"],
         "Precision": v["precision"], "Recall": v["recall"],
         "Normal": v["n_normal"], "Anomalous": v["n_anomalous"]}
        for k, v in results.items()]
rows.append({"Category": "MEAN", "AUROC": mean_auroc, "F1": mean_f1})
df = pd.DataFrame(rows)
print("\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.1%}" if isinstance(x, float) and x <= 1 else f"{x}"))

# Save CSV
df.to_csv("clip_baseline_results.csv", index=False)
print("\nSaved: clip_baseline_results.csv")

# Plot
categories_sorted = sorted(results.keys(), key=lambda c: results[c]["auroc"], reverse=True)
aurocs_sorted = [results[c]["auroc"] for c in categories_sorted]
colors = ["#2ecc71" if a >= 0.85 else "#f39c12" if a >= 0.70 else "#e74c3c" for a in aurocs_sorted]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(len(categories_sorted)), aurocs_sorted, color=colors, edgecolor="white", height=0.7)
for bar, a in zip(bars, aurocs_sorted):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f"{a:.1%}", va="center", fontsize=10, fontweight="bold")
ax.axvline(x=mean_auroc, color="#3498db", linestyle="--", linewidth=2, label=f"Mean: {mean_auroc:.1%}")
ax.set_yticks(range(len(categories_sorted)))
ax.set_yticklabels(categories_sorted, fontsize=11)
ax.set_xlabel("Image-level AUROC", fontsize=12)
ax.set_title("CLIP ViT-L/14 Zero-Shot Anomaly Detection — MVTec AD", fontsize=14, fontweight="bold")
ax.set_xlim(0, 1.15)
ax.legend(fontsize=11, loc="lower right")
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("clip_baseline_auroc.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: clip_baseline_auroc.png")
