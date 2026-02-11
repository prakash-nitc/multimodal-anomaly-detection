# Multimodal Anomaly Detection using Vision-Language Models

> **M.Tech Research Project** — Investigating the application of Multimodal Large Language Models (MLLMs) for zero-shot and few-shot anomaly detection in industrial quality inspection.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Abstract

Existing anomaly detection methods primarily rely on unimodal visual features and require extensive normal training data, limiting their adaptability to new defect types and domains. This research investigates the application of Multimodal Large Language Models (MLLMs) for zero-shot and few-shot anomaly detection in industrial quality inspection. We propose a structured multimodal prompting framework that leverages both visual understanding and language-based defect descriptions to achieve robust anomaly detection without task-specific fine-tuning.

## 🏗️ Project Structure

```
multimodal-anomaly-detection/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/                        # Documentation & reports
│   ├── literature_review/       # Paper summaries & survey
│   ├── proposal/                # Research proposal
│   └── thesis/                  # Final thesis (LaTeX)
│
├── notebooks/                   # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_clip_baseline.ipynb
│   ├── 03_mllm_inference.ipynb
│   └── 04_results_analysis.ipynb
│
├── src/                         # Source code
│   ├── data/                    # Data loading & preprocessing
│   ├── models/                  # Model wrappers (CLIP, LLaVA, etc.)
│   ├── evaluation/              # Metrics & evaluation scripts
│   ├── prompts/                 # Prompt templates
│   └── visualization/           # Plotting & heatmap generation
│
├── configs/                     # Experiment configurations
│
├── scripts/                     # Automation scripts
│   ├── download_dataset.py      # Download MVTec AD dataset
│   └── run_experiments.py       # Run all experiments
│
├── results/                     # Experiment results & figures
│   ├── tables/
│   ├── figures/
│   └── logs/
│
└── references/                  # Reference papers (BibTeX)
```

## 🔬 Methodology

### Models Used
- **CLIP** (ViT-B/32, ViT-L/14) — Zero-shot image-text anomaly scoring
- **LLaVA** — Multimodal LLM for visual defect reasoning
- **Anomalib Baselines** — PatchCore, PaDiM, STFPM for comparison

### Datasets
- **MVTec AD** — 15 industrial product categories with pixel-level annotations
- **VisA** — Visual Anomaly dataset (for generalization study)

### Evaluation Metrics
- Image-level AUROC
- Pixel-level AUROC
- F1-Score
- Per-category analysis

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/multimodal-anomaly-detection.git
cd multimodal-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_dataset.py --dataset mvtec

# Run baseline experiments
python scripts/run_experiments.py --model clip --config configs/clip_baseline.yaml
```

## 📊 Results

> Results will be updated as experiments progress.

| Method | MVTec AD (AUROC) | Zero-Shot? | Training Required? |
|--------|-----------------|------------|-------------------|
| PatchCore | - | ❌ | Yes |
| PaDiM | - | ❌ | Yes |
| CLIP (ours) | - | ✅ | No |
| LLaVA (ours) | - | ✅ | No |

## 📅 Timeline

| Phase | Semester | Status |
|-------|----------|--------|
| Literature Survey & Problem Statement | Sem 2 (Current) | 🔄 In Progress |
| Core Implementation & Baseline Experiments | Sem 3 | ⏳ Upcoming |
| Ablation Studies, Analysis & Thesis Writing | Sem 4 | ⏳ Upcoming |

## 📚 Key References

1. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), ICML 2021
2. Liu et al., "Visual Instruction Tuning" (LLaVA), NeurIPS 2023
3. Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation", CVPR 2023
4. Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (PatchCore), CVPR 2022
5. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- Prof. [Name], [Department], [College Name] — Research Supervisor
- [College Name] — Computational Resources
