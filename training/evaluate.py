#!/usr/bin/env python3
"""
training/evaluate.py

Evaluates a trained YOLOv8 model checkpoint against a designated dataset split (val/test).

STRICT EVALUATION RULES:
1. TRAIN AND TUNE ON TRAIN/VALID SPLITS ONLY.
2. RUN THE TEST SPLIT ONCE AT THE VERY END FOR THE FINAL REPORT.
3. Report precision, recall, mAP@50, mAP@50-95 overall and per-class,
   the confusion matrix, and inference latency per image.

Usage:
    # Validation evaluation (during tuning/validation)
    python training/evaluate.py --weights runs/baseline_v1_yolov8n/weights/best.pt --split val

    # Final evaluation (ONLY once after all model iterations are frozen)
    python training/evaluate.py --weights runs/baseline_v1_yolov8n/weights/best.pt --split test --allow-test
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


def resolve_device(device_setting: str) -> str:
    if device_setting == "auto" or device_setting is None:
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device_setting


def evaluate_model(
    weights_path: Path,
    data_yaml_path: Path,
    split: str = "val",
    device: str = "auto",
    allow_test: bool = False,
    output_dir: str = "results/metrics",
):
    print("=" * 70)
    print(" AquaVigil - Model Evaluation")
    print("=" * 70)

    # Enforce evaluation rule: Test split only run once with explicit confirmation
    if split == "test" and not allow_test:
        print("[!] EVALUATION PROTOCOL SAFEGUARD TRIGGERED:")
        print("    Evaluating on the test split should only be executed ONCE at the very end")
        print("    of the project to avoid data leakage and benchmark gaming.")
        print("    To run test evaluation, re-run with the flag: --allow-test")
        sys.exit(1)

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

    target_device = resolve_device(device)

    print(f"Loading Model:  {weights_path}")
    print(f"Dataset YAML:   {data_yaml_path}")
    print(f"Split:          {split}")
    print(f"Device:         {target_device} (requested: {device})\n")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml_path.resolve()),
        split=split,
        device=target_device,
        project=output_dir,
        name=f"eval_{split}",
        plots=True,
        exist_ok=True,
    )

    # Extract overall metrics
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # Extract per-class metrics
    class_names = metrics.names if hasattr(metrics, "names") else {}
    per_class_metrics = {}

    p_curve = getattr(metrics.box, "p", None)
    r_curve = getattr(metrics.box, "r", None)
    ap50 = getattr(metrics.box, "ap50", None)
    ap = getattr(metrics.box, "ap", None)

    for idx, c_name in class_names.items():
        per_class_metrics[c_name] = {
            "precision": float(p_curve[idx]) if p_curve is not None and len(p_curve) > idx else None,
            "recall": float(r_curve[idx]) if r_curve is not None and len(r_curve) > idx else None,
            "mAP@50": float(ap50[idx]) if ap50 is not None and len(ap50) > idx else None,
            "mAP@50-95": float(ap[idx]) if ap is not None and len(ap) > idx else None,
        }

    # Speed metrics
    speed = getattr(metrics, "speed", {})
    inference_time_ms = speed.get("inference", 0.0)
    preprocess_time_ms = speed.get("preprocess", 0.0)
    postprocess_time_ms = speed.get("postprocess", 0.0)
    total_latency_ms = preprocess_time_ms + inference_time_ms + postprocess_time_ms

    # Confusion matrix extraction
    confusion_matrix_data = None
    if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
        matrix = getattr(metrics.confusion_matrix, "matrix", None)
        if matrix is not None:
            confusion_matrix_data = matrix.tolist()

    eval_results = {
        "split": split,
        "weights": str(weights_path),
        "dataset_yaml": str(data_yaml_path),
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mAP_50": round(map50, 4),
            "mAP_50_95": round(map50_95, 4),
        },
        "per_class": per_class_metrics,
        "speed_ms_per_image": {
            "preprocess": round(preprocess_time_ms, 2),
            "inference": round(inference_time_ms, 2),
            "postprocess": round(postprocess_time_ms, 2),
            "total": round(total_latency_ms, 2),
        },
        "confusion_matrix": confusion_matrix_data,
        "plots_saved_to": str(Path(metrics.save_dir).resolve()),
    }

    out_file = Path(output_dir) / f"evaluation_{split}_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # Print clean summary table
    print("\n" + "=" * 70)
    print(f" EVALUATION SUMMARY ({split.upper()} SPLIT)")
    print("=" * 70)
    print(f"  Precision:           {precision:.4f}")
    print(f"  Recall:              {recall:.4f}")
    print(f"  mAP@50:              {map50:.4f}")
    print(f"  mAP@50-95:           {map50_95:.4f}")
    print(f"  Inference Latency:   {inference_time_ms:.2f} ms / image (Total: {total_latency_ms:.2f} ms)")
    print("-" * 70)
    print(f"  {'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP@50':<12} {'mAP@50-95':<12}")
    print("-" * 70)
    for c_name, scores in per_class_metrics.items():
        p_val = f"{scores['precision']:.4f}" if scores['precision'] is not None else "N/A"
        r_val = f"{scores['recall']:.4f}" if scores['recall'] is not None else "N/A"
        map50_val = f"{scores['mAP@50']:.4f}" if scores['mAP@50'] is not None else "N/A"
        map_val = f"{scores['mAP@50-95']:.4f}" if scores['mAP@50-95'] is not None else "N/A"
        print(f"  {c_name:<20} {p_val:<12} {r_val:<12} {map50_val:<12} {map_val:<12}")
    print("-" * 70)
    print(f"[+] Confusion matrix and evaluation plots: {metrics.save_dir}")
    print(f"[+] Full JSON evaluation report saved to:   {out_file.resolve()}\n")
    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AquaVigil YOLOv8 fish disease detection model.")
    parser.add_argument("--weights", type=str, default="runs/baseline_v1_yolov8n/weights/best.pt", help="Path to weights (.pt)")
    parser.add_argument("--data", type=str, default="dataset/processed/baseline_v1/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cpu', '0')")
    parser.add_argument("--allow-test", action="store_true", help="Explicit confirmation to evaluate on the test split")
    parser.add_argument("--output-dir", type=str, default="results/metrics", help="Directory to save evaluation results")
    args = parser.parse_args()

    evaluate_model(
        weights_path=Path(args.weights),
        data_yaml_path=Path(args.data),
        split=args.split,
        device=args.device,
        allow_test=args.allow_test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
