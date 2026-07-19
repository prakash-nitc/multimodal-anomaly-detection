# ============================================================
# DA-ZVAD — FULL SEM-3 EXPERIMENT RUN (one Kaggle GPU session)
# ============================================================
# Supersedes the Phase-0 probe: one run produces every table the
# September review needs. All stages are cached and RESUMABLE —
# if the session dies, rerun this cell and it continues.
#
# HOW TO RUN (Kaggle):
#   1. New Notebook -> Settings -> Accelerator = GPU (T4/P100).
#   2. Add Input -> Datasets:
#        - REQUIRED: a ShanghaiTech dataset (testing/frames + test_frame_mask)
#        - OPTIONAL: MVTec AD; CUHK Avenue (testing_videos + labels .mat)
#   3. Paste this whole file as one cell. Run. (~1-3 h depending
#      on datasets present; ShanghaiTech-only ≈ 60-90 min.)
#   4. Download /kaggle/working/results_bundle.zip and send it back.
#
# Set RUN_EXPLANATIONS = True only if you also want the LLaVA
# explanation gallery in this session (adds model download ~4 GB
# and ~15 min; can be a separate session instead).
# ============================================================

RUN_EXPLANATIONS = False
FRAME_STEP = 2          # score every 2nd frame (GT subsampled identically)
WINDOWS = [1, 5, 9, 15]

import glob, os, shutil, subprocess, sys

# --- 1. environment -----------------------------------------------------
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "open-clip-torch", "opencv-python-headless"], check=True)
if RUN_EXPLANATIONS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers", "accelerate", "bitsandbytes"], check=True)

# --- 2. get the framework (public repo, dev branch) ---------------------
REPO = "/kaggle/working/mad"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", "dev",
                    "https://github.com/prakash-nitc/multimodal-anomaly-detection",
                    REPO], check=True)
os.chdir(REPO)
sys.path.insert(0, REPO)

from da_zvad.config import DAZVADConfig
from da_zvad.grid import DatasetSpec, run_grid
from da_zvad.context_sweep import run_context_sweep

# --- 3. locate datasets under /kaggle/input -----------------------------
def locate(candidates):
    """First /kaggle/input entry (or its direct child) containing all candidate dirs."""
    for base in sorted(glob.glob("/kaggle/input/*")):
        for sub in ["", *os.listdir(base)]:
            root = os.path.join(base, sub) if sub else base
            if not os.path.isdir(root):
                continue
            if all(os.path.isdir(os.path.join(root, c)) for c in candidates):
                return root
    return None

specs = []
sh_root = (locate(["testing/frames"]) or locate(["frames"])
           or locate(["testing_frames"]))
if sh_root:
    print(f"[data] ShanghaiTech: {sh_root}")
    specs.append(DatasetSpec("shanghaitech", sh_root, domain="surveillance",
                             description="a university campus walkway with pedestrians"))
else:
    print("[data] ShanghaiTech NOT FOUND — add a dataset with testing/frames + test_frame_mask")

av_root = locate(["testing_videos"]) or locate(["testing/videos"])
if av_root:
    print(f"[data] Avenue: {av_root}")
    specs.append(DatasetSpec("avenue", av_root, domain="surveillance",
                             description="a subway station entrance with commuters"))

mv_root = None
for base in sorted(glob.glob("/kaggle/input/*")):
    for sub in ["", "mvtec-ad", "mvtec_anomaly_detection"]:
        root = os.path.join(base, sub) if sub else base
        if os.path.isdir(os.path.join(root, "bottle", "test")):
            mv_root = root
            break
if mv_root:
    print(f"[data] MVTec: {mv_root}")
    for cat in ["bottle", "carpet", "screw", "transistor"]:   # 2 textures + 2 objects
        specs.append(DatasetSpec("mvtec", mv_root, category=cat, domain="industrial",
                                 description=f"an industrial quality-inspection image of a {cat}"))

assert specs, "No datasets found — add at least ShanghaiTech via Add Input."

# --- 4. run everything (cached; safe to rerun) --------------------------
base = DAZVADConfig(clip_model="ViT-L-14", frame_step=FRAME_STEP)

print("\n================ GRID (context on/off x windows) ================", flush=True)
run_grid(specs, windows=WINDOWS, base=base, out_dir="results")

print("\n================ CONTEXT SWEEP (none/generic/matched/mismatched) ================", flush=True)
run_context_sweep(specs, windows=[1, 5], base=base, out_dir="results")

if RUN_EXPLANATIONS and sh_root:
    print("\n================ EXPLANATION GALLERY (matched vs mismatched) ================", flush=True)
    from da_zvad.explain_shift import run_explain_shift
    llava_base = DAZVADConfig(clip_model="ViT-L-14", frame_step=FRAME_STEP,
                              reasoner="llava")
    run_explain_shift(specs[0], base=llava_base, max_events=6, out_dir="results")

# --- 5. bundle outputs for download -------------------------------------
bundle = "/kaggle/working/results_bundle"
shutil.rmtree(bundle, ignore_errors=True)
shutil.copytree("results/tables", bundle + "/tables", dirs_exist_ok=True)
if os.path.isdir("results/explanations"):
    shutil.copytree("results/explanations", bundle + "/explanations", dirs_exist_ok=True)
shutil.make_archive(bundle, "zip", bundle)
print(f"\nDONE. Download results_bundle.zip from /kaggle/working and send it back.")
