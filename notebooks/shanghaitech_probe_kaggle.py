# ============================================================
# PHASE-0 DE-RISK PROBE — Zero-shot CLIP on ShanghaiTech
# ============================================================
# Purpose: measure where naive training-free CLIP scoring stands
# on a real surveillance benchmark (frame-level AUROC).
# Decision gate: >= ~0.75 -> detection can co-headline;
#                ~0.60s   -> explanation-led framing (plan continues either way).
#
# HOW TO RUN (Kaggle):
#   1. New Notebook -> Settings: Accelerator = GPU (T4/P100).
#   2. Add data -> search "shanghaitech" (need testing frames +
#      test_frame_mask .npy ground truth).
#   3. Paste this whole file as one cell. Run.
#   Runtime: ~20-40 min at FRAME_STEP=2 on a T4.
#
# One GPU pass -> image features are cached, so we get a mini-
# ablation for free: 3 prompt variants x 4 smoothing windows.
# ============================================================

# !pip install open-clip-torch -q

import os, glob, json, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# ---------------- config ----------------
MODEL, PRETRAINED = "ViT-L-14", "laion2b_s32b_b82k"
FRAME_STEP = 2            # score every 2nd frame (GT subsampled identically)
BATCH = 64
WINDOWS = [1, 5, 9, 15]   # temporal smoothing sweep (1 = off)
OUT_DIR = "/kaggle/working"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

# ---------------- locate dataset ----------------
def find_shanghaitech():
    cands = []
    for base in glob.glob("/kaggle/input/*"):
        for sub in ["", "shanghaitech", "ShanghaiTech"]:
            root = os.path.join(base, sub) if sub else base
            for f in ["testing/frames", "frames", "testing_frames"]:
                if os.path.isdir(os.path.join(root, f)):
                    cands.append((root, os.path.join(root, f)))
    if not cands:
        raise FileNotFoundError(
            "ShanghaiTech not found under /kaggle/input. Add a dataset that "
            "contains testing/frames and testing/test_frame_mask.")
    return cands[0]

ROOT, FRAMES = find_shanghaitech()
MASKS = next((p for p in [os.path.join(ROOT, "testing/test_frame_mask"),
                          os.path.join(ROOT, "test_frame_mask"),
                          os.path.join(ROOT, "testing/frame_masks")]
              if os.path.isdir(p)), None)
print("frames:", FRAMES)
print("masks :", MASKS)
assert MASKS, "Ground-truth .npy masks not found — AUROC needs them."

# ---------------- model ----------------
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(MODEL)

@torch.no_grad()
def text_ensemble(prompts):
    tok = tokenizer(prompts).to(DEVICE)
    f = F.normalize(model.encode_text(tok), dim=-1)
    return F.normalize(f.mean(0, keepdim=True), dim=-1)

# ---- prompt variants (mirrors da_zvad.prompts + M3 grounding) ----
GENERIC = (["a normal scene", "a typical everyday scene with nothing unusual",
            "a calm and ordinary situation"],
           ["an abnormal scene", "an unusual or unexpected event",
            "something is clearly wrong"])
SURV = (["a normal day with people behaving ordinarily",
         "a calm and safe public area",
         "nothing dangerous or abnormal is happening"],
        ["a dangerous or violent event",
         "a fight, robbery or accident in progress",
         "an abnormal and unsafe situation"])
CTX = "a university campus walkway with pedestrians"
SURV_CTX = (SURV[0] + [f"{CTX}, everything is normal",
                       f"{CTX}, a usual and safe moment"],
            SURV[1] + [f"{CTX}, but something abnormal is happening",
                       f"{CTX}, but a dangerous or unexpected event occurs"])

VARIANTS = {"generic": GENERIC, "surveillance": SURV, "surveillance+context": SURV_CTX}
TEXT = {k: (text_ensemble(n), text_ensemble(a)) for k, (n, a) in VARIANTS.items()}

# ---------------- scoring ----------------
@torch.no_grad()
def image_features(paths):
    feats = []
    for i in range(0, len(paths), BATCH):
        ims = [preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+BATCH]]
        x = torch.stack(ims).to(DEVICE)
        feats.append(F.normalize(model.encode_image(x), dim=-1).cpu())
    return torch.cat(feats)

def scores_from_feats(feats, normal_e, abnormal_e):
    text = torch.cat([normal_e, abnormal_e]).cpu().float()
    sim = feats.float() @ text.T * model.logit_scale.exp().cpu().float()
    return sim.softmax(dim=-1)[:, 1].numpy()

def smooth(s, w):
    if w <= 1: return s
    pad = w // 2
    return np.convolve(np.pad(s, pad, mode="edge"), np.ones(w)/w, "same")[pad:pad+len(s)]

def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else np.nan

# ---------------- run ----------------
clips = sorted(d for d in os.listdir(FRAMES) if os.path.isdir(os.path.join(FRAMES, d)))
print(f"{len(clips)} test clips, FRAME_STEP={FRAME_STEP}")

per_clip, all_scores, all_labels = [], {k: {w: [] for w in WINDOWS} for k in VARIANTS}, []
t0 = time.time()
for clip in tqdm(clips, desc="clips"):
    paths = sorted(glob.glob(os.path.join(FRAMES, clip, "*.jpg")) +
                   glob.glob(os.path.join(FRAMES, clip, "*.png")))
    gt_path = os.path.join(MASKS, clip + ".npy")
    if not paths or not os.path.isfile(gt_path):
        warnings.warn(f"skipping {clip} (missing frames or GT)"); continue
    labels = np.load(gt_path).astype(int).ravel()
    n = min(len(paths), len(labels))
    idx = np.arange(0, n, FRAME_STEP)               # SAME indices for frames & GT
    paths, labels = [paths[i] for i in idx], labels[idx]

    feats = image_features(paths)
    row = {"clip": clip, "frames": len(paths), "anomalous": int(labels.sum())}
    for k in VARIANTS:
        raw = scores_from_feats(feats, *TEXT[k])
        for w in WINDOWS:
            s = smooth(raw, w)
            all_scores[k][w].append(s)
            if k == "surveillance+context" and w == 5:
                row["auroc_ctx_w5"] = auroc(s, labels)
    all_labels.append(labels)
    per_clip.append(row)

labels_cat = np.concatenate(all_labels)
print(f"\nscored {labels_cat.size} frames in {(time.time()-t0)/60:.1f} min "
      f"({100*labels_cat.mean():.1f}% anomalous)")

# ---------------- results table ----------------
print("\n===== FRAME-LEVEL AUROC (micro, concatenated) =====")
print(f"{'prompt variant':<24}" + "".join(f"  w={w:<4}" for w in WINDOWS))
results = {}
for k in VARIANTS:
    row = [auroc(np.concatenate(all_scores[k][w]), labels_cat) for w in WINDOWS]
    results[k] = dict(zip(map(str, WINDOWS), map(float, row)))
    print(f"{k:<24}" + "".join(f"  {v:.3f} " for v in row))

best = max((v, k, w) for k, d in results.items() for w, v in d.items())
print(f"\nBEST: {best[0]:.3f}  ({best[1]}, window={best[2]})")
print("GATE: >=0.75 detection co-headline | ~0.60s explanation-led framing")

# ---------------- save ----------------
import csv
with open(os.path.join(OUT_DIR, "probe_per_clip.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_clip[0].keys())); w.writeheader(); w.writerows(per_clip)
with open(os.path.join(OUT_DIR, "probe_results.json"), "w") as f:
    json.dump({"model": MODEL, "pretrained": PRETRAINED, "frame_step": FRAME_STEP,
               "n_frames": int(labels_cat.size), "auroc": results}, f, indent=2)
print("saved: probe_per_clip.csv, probe_results.json  — paste both back to Claude")
