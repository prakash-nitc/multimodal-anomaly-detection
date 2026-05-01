"""
CLIP Zero-Shot Baseline for MVTec AD
=====================================
Main script to run zero-shot anomaly detection using CLIP on all 15
MVTec AD categories and produce per-category AUROC results.

Usage:
    python scripts/run_clip_baseline.py --data_root <path_to_mvtec>
    python scripts/run_clip_baseline.py --data_root <path> --model ViT-B-32 --pretrained laion2b_s34b_b79k
    python scripts/run_clip_baseline.py --data_root <path> --categories bottle cable capsule
"""

import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mvtec_dataset import MVTecDataset, MVTEC_CATEGORIES
from src.prompts.anomaly_prompts import get_prompt_pair
from src.models.clip_zero_shot import CLIPZeroShotDetector
from src.evaluation.metrics import compute_auroc, compute_optimal_f1, format_results_table
from src.visualization.plot_results import plot_category_auroc


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Zero-Shot Anomaly Detection on MVTec AD")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to MVTec AD root directory (containing bottle/, cable/, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14",
        help="OpenCLIP model name (default: ViT-L-14)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s32b_b82k",
        help="Pretrained weights (default: laion2b_s32b_b82k)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific categories to evaluate (default: all 15)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results (default: results/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate data root
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: Data root does not exist: {data_root}")
        print("Please download MVTec AD from: https://www.mvtec.com/company/research/datasets/mvtec-ad")
        sys.exit(1)

    # Categories to evaluate
    categories = args.categories if args.categories else MVTEC_CATEGORIES

    # Validate categories
    for cat in categories:
        if cat not in MVTEC_CATEGORIES:
            print(f"ERROR: Unknown category '{cat}'. Valid categories: {MVTEC_CATEGORIES}")
            sys.exit(1)
        if not (data_root / cat).exists():
            print(f"WARNING: Category directory not found: {data_root / cat}. Skipping.")
            categories = [c for c in categories if c != cat]

    if not categories:
        print("ERROR: No valid categories found.")
        sys.exit(1)

    print("=" * 60)
    print("CLIP Zero-Shot Anomaly Detection — MVTec AD")
    print("=" * 60)
    print(f"Model:      {args.model} ({args.pretrained})")
    print(f"Data root:  {data_root}")
    print(f"Categories: {len(categories)}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Load model
    detector = CLIPZeroShotDetector(
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
    )

    # Run evaluation for each category
    results = {}
    total_start = time.time()

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] Evaluating: {category}")
        print("-" * 40)

        # Create dataset and dataloader
        dataset = MVTecDataset(
            root_dir=str(data_root),
            category=category,
            split="test",
            transform=detector.get_preprocess(),
        )
        print(f"  {dataset}")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,  # Windows compatibility
            pin_memory=False,
        )

        # Get prompts for this category
        normal_prompts, abnormal_prompts = get_prompt_pair(category)

        # Compute anomaly scores
        cat_start = time.time()
        scores, labels = detector.compute_anomaly_scores(dataloader, normal_prompts, abnormal_prompts)
        cat_time = time.time() - cat_start

        # Compute metrics
        auroc = compute_auroc(scores, labels)
        f1_result = compute_optimal_f1(scores, labels)

        results[category] = {
            "auroc": auroc,
            "f1": f1_result["f1"],
            "precision": f1_result["precision"],
            "recall": f1_result["recall"],
            "threshold": f1_result["threshold"],
            "num_normal": dataset.num_normal,
            "num_anomalous": dataset.num_anomalous,
            "time_sec": cat_time,
        }

        print(f"  AUROC:     {auroc:.1%}")
        print(f"  F1:        {f1_result['f1']:.1%} (P={f1_result['precision']:.1%}, R={f1_result['recall']:.1%})")
        print(f"  Time:      {cat_time:.1f}s")

    total_time = time.time() - total_start

    # Compute mean results
    valid_results = {k: v for k, v in results.items() if not np.isnan(v["auroc"])}
    results["MEAN"] = {
        "auroc": np.mean([v["auroc"] for v in valid_results.values()]),
        "f1": np.mean([v["f1"] for v in valid_results.values()]),
    }

    # Print summary table
    print("\n")
    print(format_results_table(results))
    print(f"\nTotal time: {total_time:.1f}s")

    # Save results
    output_dir = Path(args.output_dir)

    # Save CSV
    csv_path = output_dir / "tables" / "clip_baseline_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cat, metrics in results.items():
        rows.append({"category": cat, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save plot
    fig_path = output_dir / "figures" / "clip_baseline_auroc.png"
    plot_category_auroc(results, str(fig_path))

    print(f"\n{'=' * 60}")
    print(f"Done! Mean AUROC: {results['MEAN']['auroc']:.1%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
