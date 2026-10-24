#!/usr/bin/env python3
"""
training/train.py

Initiates YOLOv8 object detection model training using the Ultralytics Python API.
Reads hyperparameter and dataset configuration from configs/model_config.yaml.

Usage:
    python training/train.py
    python training/train.py --config configs/model_config.yaml --device cpu
    python training/train.py --config configs/model_config.yaml --device 0
"""

import argparse
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


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is not installed. Run 'pip install PyYAML'.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_prerequisites(config: dict) -> tuple[bool, str]:
    # 1. Ultralytics check
    if YOLO is None:
        return False, "Ultralytics is not installed. Install requirements first via 'pip install ultralytics'."

    # 2. Dataset YAML check
    data_yaml_str = config.get("data", {}).get("yaml_path")
    if not data_yaml_str:
        return False, "Configuration missing 'data.yaml_path' field in model_config.yaml."

    data_yaml_path = Path(data_yaml_str)
    if not data_yaml_path.exists():
        return False, (
            f"Dataset configuration YAML does not exist: {data_yaml_path.resolve()}\n"
            "Please extract the Roboflow dataset into 'dataset/raw/' and ensure 'data.yaml' is present."
        )

    # 3. Check data.yaml contents
    try:
        dataset_meta = load_yaml(data_yaml_path)
    except Exception as e:
        return False, f"Failed to parse dataset YAML ({data_yaml_path}): {e}"

    train_path = dataset_meta.get("train")
    if not train_path:
        return False, f"Dataset YAML ({data_yaml_path}) does not define a 'train' path."

    val_path = dataset_meta.get("val") or dataset_meta.get("valid")
    if not val_path:
        print("[WARNING] Dataset YAML does not specify a validation set ('val'). Training without validation may result in overfitting or evaluation failures.")

    return True, ""


def resolve_device(device_setting: str) -> str:
    if device_setting == "auto" or device_setting is None:
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device_setting


def run_training(config_path: Path, device_override: str = None, smoke: bool = False):
    print("=" * 70)
    print(" AquaVigil - Model Training" + (" [SMOKE TEST]" if smoke else ""))
    print("=" * 70)

    if not config_path.exists():
        print(f"[!] Config file not found: {config_path.resolve()}")
        sys.exit(1)

    config = load_yaml(config_path)

    # Validate prerequisites
    ok, err_msg = validate_prerequisites(config)
    if not ok:
        print(f"\n[!] PREREQUISITE ERROR:\n{err_msg}\n")
        print("Training cannot start until requirements and dataset are verified.")
        sys.exit(1)

    model_weights = config.get("model", {}).get("weights", "yolov8n.pt")
    data_yaml = config.get("data", {}).get("yaml_path")
    training_args = config.get("training", {})
    output_args = config.get("output", {})

    raw_device = device_override or training_args.get("device", "auto")
    device = resolve_device(raw_device)

    epochs = 1 if smoke else training_args.get("epochs", 50)
    fraction = 0.05 if smoke else 1.0
    run_name = (output_args.get("name", "baseline_v1_yolov8n") + "_smoke") if smoke else output_args.get("name", "baseline_v1_yolov8n")
    project = output_args.get("project", "runs")

    print("\nTraining Parameters:")
    print(f"  Model Weights:   {model_weights}")
    print(f"  Dataset Config:  {data_yaml}")
    print(f"  Epochs:          {epochs}" + (" (Smoke test: 1 epoch)" if smoke else ""))
    print(f"  Data Fraction:   {fraction}" + (" (Smoke test: 5% data)" if smoke else ""))
    print(f"  Batch Size:      {training_args.get('batch', 16)}")
    print(f"  Image Size:      {training_args.get('imgsz', 640)}")
    print(f"  Config Device:   {raw_device} -> Resolved: {device}")
    print(f"  Output Project:  {project}")
    print(f"  Run Name:        {run_name}\n")

    print("[*] Initializing YOLO model...")
    model = YOLO(model_weights)

    train_kwargs = {
        "data": str(Path(data_yaml).resolve()),
        "epochs": epochs,
        "batch": training_args.get("batch", 16),
        "imgsz": training_args.get("imgsz", 640),
        "device": device,
        "workers": training_args.get("workers", 2),
        "seed": training_args.get("seed", 42),
        "project": project,
        "name": run_name,
        "exist_ok": True if smoke else output_args.get("exist_ok", False),
        "save_period": output_args.get("save_period", -1),
        "patience": training_args.get("patience", 15),
    }

    if smoke:
        train_kwargs["fraction"] = 0.05

    print("[*] Launching training process...")
    results = model.train(**train_kwargs)

    print("\n[+] Training complete.")
    save_dir = getattr(results, "save_dir", Path(project) / run_name)
    print(f"Best weights saved at: {save_dir}/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train AquaVigil YOLOv8 fish disease detection model.")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model configuration YAML")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cpu', '0', '1', etc.)")
    parser.add_argument("--smoke", action="store_true", help="Run a smoke test (1 epoch, fraction=0.05)")
    args = parser.parse_args()

    run_training(Path(args.config), args.device, args.smoke)


if __name__ == "__main__":
    main()

