#!/usr/bin/env python3
"""
training/evaluate.py

Evaluates a trained YOLOv8 model checkpoint against a designated dataset split (val/test).
Computes precision, recall, mAP@50, mAP@50-95, and saves results to results/metrics/.

Usage:
    python training/evaluate.py --weights models/best.pt --data dataset/raw/data.yaml --split val
    python training/evaluate.py --weights runs/detect/aquavigil_v8_baseline/weights/best.pt --split test
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def evaluate_model(weights_path: Path, data_yaml_path: Path, split: str = "val", device: str = "cpu"):
    print("=" * 70)
    print(" AquaVigil - Model Evaluation")
    print("=" * 70)

    # 1. Dependency check
    if YOLO is None:
        print("[!] Error: Ultralytics is not installed. Run 'pip install ultralytics'.")
        sys.exit(1)

    # 2. Weights check
    if not weights_path.exists():
        print(f"[!] Error: Model weights not found at: {weights_path.resolve()}")
        print("    A trained model checkpoint (.pt) is required for evaluation.")
        print("    Please train a model first using: python training/train.py")
        sys.exit(1)

    # 3. Data YAML check
    if not data_yaml_path.exists():
        print(f"[!] Error: Dataset configuration YAML not found at: {data_yaml_path.resolve()}")
        sys.exit(1)

    print(f"Loading Model: {weights_path}")
    print(f"Dataset YAML:  {data_yaml_path}")
    print(f"Split:         {split}")
    print(f"Device:        {device}\n")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml_path.resolve()),
        split=split,
        device=device,
        project="results/metrics",
        name=f"eval_{split}",
        exist_ok=True,
    )

    results_dict = {
        "split": split,
        "weights": str(weights_path),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP_50": float(metrics.box.map50),
        "mAP_50_95": float(metrics.box.map),
    }

    metrics_out = Path("results/metrics") / f"metrics_{split}.json"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    print("\n--- Evaluation Summary ---")
    print(f"  Precision: {results_dict['precision']:.4f}")
    print(f"  Recall:    {results_dict['recall']:.4f}")
    print(f"  mAP@50:    {results_dict['mAP_50']:.4f}")
    print(f"  mAP@50-95: {results_dict['mAP_50_95']:.4f}")
    print(f"\n[+] Detailed metrics exported to: {metrics_out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AquaVigil YOLOv8 fish disease detection model.")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained weights file (.pt)")
    parser.add_argument("--data", type=str, default="dataset/raw/data.yaml", help="Path to dataset configuration YAML")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run evaluation on ('cpu' or '0')")
    args = parser.parse_args()

    evaluate_model(Path(args.weights), Path(args.data), args.split, args.device)


if __name__ == "__main__":
    main()
