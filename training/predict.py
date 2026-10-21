#!/usr/bin/env python3
"""
training/predict.py

Runs inference on a user-provided image using a trained YOLOv8 model checkpoint.
Draws detected bounding boxes, class labels, and confidence percentages,
and saves the annotated output to results/predictions/.

Usage:
    python training/predict.py --weights models/best.pt --source sample.jpg
    python training/predict.py --weights runs/detect/aquavigil_v8_baseline/weights/best.pt --source sample.jpg --conf 0.3
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def run_prediction(weights_path: Path, source_path: Path, conf: float = 0.25, iou: float = 0.45, device: str = "cpu", output_dir: str = "results/predictions"):
    print("=" * 70)
    print(" AquaVigil - Model Inference")
    print("=" * 70)

    if YOLO is None:
        print("[!] Error: Ultralytics is not installed. Run 'pip install ultralytics'.")
        sys.exit(1)

    if not weights_path.exists():
        print(f"[!] Error: Model weights not found at: {weights_path.resolve()}")
        print("    Inference requires a trained checkpoint (.pt).")
        print("    Please train a model first using: python training/train.py")
        sys.exit(1)

    if not source_path.exists():
        print(f"[!] Error: Target image file not found at: {source_path.resolve()}")
        sys.exit(1)

    print(f"Loading Weights: {weights_path}")
    print(f"Input Source:    {source_path}")
    print(f"Confidence:      {conf}")
    print(f"Device:          {device}\n")

    model = YOLO(str(weights_path))

    results = model.predict(
        source=str(source_path.resolve()),
        conf=conf,
        iou=iou,
        device=device,
        project=output_dir,
        name="single_prediction",
        save=True,
        exist_ok=True,
    )

    print("\n--- Detection Results ---")
    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            print("  No disease symptoms detected above the confidence threshold.")
        else:
            print(f"  Detected {len(boxes)} object(s):")
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id] if model.names else str(cls_id)
                confidence = float(box.conf[0])
                coords = [round(float(c), 2) for c in box.xyxy[0]]
                print(f"    - {cls_name} ({confidence * 100:.1f}%) at [x1={coords[0]}, y1={coords[1]}, x2={coords[2]}, y2={coords[3]}]")

    print(f"\n[+] Annotated output saved to: {results[0].save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run AquaVigil YOLOv8 inference on fish images.")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained model weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to target image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or '0')")
    parser.add_argument("--output", type=str, default="results/predictions", help="Output directory")
    args = parser.parse_args()

    run_prediction(Path(args.weights), Path(args.source), args.conf, args.iou, args.device, args.output)


if __name__ == "__main__":
    main()
