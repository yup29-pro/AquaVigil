#!/usr/bin/env python3
"""
scripts/visualize_annotations.py

Visualizes YOLO axis-aligned bounding boxes overlaid onto fish disease images.
Accepts an image path (or directory) and an optional label file or directory,
and saves the annotated output to results/predictions/ or a user-specified output file.

Expected YOLO Annotation Format:
    <class_id> <x_center> <y_center> <width> <height>
    (All coordinates normalized to [0.0, 1.0])

Usage:
    python scripts/visualize_annotations.py --image dataset/raw/train/images/fish_01.jpg
    python scripts/visualize_annotations.py --image dataset/raw/train/images/fish_01.jpg --label dataset/raw/train/labels/fish_01.txt --config dataset/raw/data.yaml
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import yaml
except ImportError:
    yaml = None

# Fallback distinct color palette (BGR format for OpenCV)
DISTINCT_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Olive
    (128, 128, 0),  # Teal
]


def load_class_names(config_path: Path) -> list:
    if not config_path or not config_path.exists() or yaml is None:
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        return names
    except Exception:
        return []


def parse_yolo_labels(label_path: Path) -> list:
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                print(f"[!] Warning: Line {line_no} in {label_path.name} is malformed (expected 5 items, got {len(parts)}). Skipping line.")
                continue
            try:
                cls_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append((cls_id, xc, yc, w, h))
            except ValueError as e:
                print(f"[!] Warning: Line {line_no} could not be parsed: {e}. Skipping.")
    return boxes


def draw_annotations(image, boxes: list, class_names: list):
    img_h, img_w = image.shape[:2]

    for (cls_id, xc, yc, w, h) in boxes:
        # Convert normalized coordinates to absolute pixels
        box_w = int(w * img_w)
        box_h = int(h * img_h)
        x1 = max(0, int((xc - w / 2.0) * img_w))
        y1 = max(0, int((yc - h / 2.0) * img_h))
        x2 = min(img_w - 1, x1 + box_w)
        y2 = min(img_h - 1, y1 + box_h)

        color = DISTINCT_COLORS[cls_id % len(DISTINCT_COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label tag text
        name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        label_text = f"{name} (#{cls_id})"

        # Label background pill
        (txt_w, txt_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_y1 = max(0, y1 - txt_h - 6)
        tag_y2 = y1
        cv2.rectangle(image, (x1, tag_y1), (x1 + txt_w + 6, tag_y2), color, -1)
        cv2.putText(
            image,
            label_text,
            (x1 + 3, tag_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image


def draw_with_pil(image_path: Path, boxes: list, class_names: list, output_path: Path):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        img_w, img_h = im.size

        # Distinct RGB colors
        rgb_palette = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 140, 0),  # Orange
            (148, 0, 211),  # Purple
        ]

        for (cls_id, xc, yc, w, h) in boxes:
            box_w = w * img_w
            box_h = h * img_h
            x1 = max(0, int((xc - w / 2.0) * img_w))
            y1 = max(0, int((yc - h / 2.0) * img_h))
            x2 = min(img_w - 1, int(x1 + box_w))
            y2 = min(img_h - 1, int(y1 + box_h))

            color = rgb_palette[cls_id % len(rgb_palette)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            label_text = f"{name} (#{cls_id})"

            # Draw tag banner
            tag_y1 = max(0, y1 - 18)
            draw.rectangle([x1, tag_y1, x1 + len(label_text) * 8 + 8, y1], fill=color)
            draw.text((x1 + 4, tag_y1 + 2), label_text, fill=(255, 255, 255))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(output_path, quality=95)
        print(f"[+] Annotated image successfully saved (via PIL) to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO axis-aligned annotations overlaid on an image."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--label", type=str, default=None, help="Path to label .txt file (defaults to matching image name in labels/ or adjacent)")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to config YAML containing class names")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated image (default: results/predictions/<image_name>_annotated.jpg)")
    args = parser.parse_args()

    if cv2 is None and Image is None:
        print("[!] Error: Neither OpenCV nor Pillow is installed. Please install 'pillow' or 'opencv-python'.")
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[!] Error: Image not found: {image_path.resolve()}")
        sys.exit(1)

    # Determine label path
    if args.label:
        label_path = Path(args.label)
    else:
        parent_dir = image_path.parent
        if parent_dir.name == "images":
            candidate = parent_dir.parent / "labels" / f"{image_path.stem}.txt"
        else:
            candidate = parent_dir / f"{image_path.stem}.txt"
        label_path = candidate

    print(f"Image: {image_path}")
    print(f"Label: {label_path} (exists: {label_path.exists()})")

    # Load classes
    class_names = []
    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                model_cfg = yaml.safe_load(f)
            data_yaml = model_cfg.get("data", {}).get("yaml_path")
            if data_yaml and Path(data_yaml).exists():
                class_names = load_class_names(Path(data_yaml))
        except Exception:
            pass
        if not class_names:
            class_names = load_class_names(cfg_path)

    boxes = parse_yolo_labels(label_path)
    print(f"Found {len(boxes)} annotation(s).")

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path("results/predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{image_path.stem}_annotated.jpg"

    if cv2 is not None:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[!] Error: Unable to read image with OpenCV: {image_path}")
            sys.exit(1)
        annotated = draw_annotations(image, boxes, class_names)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[+] Annotated image successfully saved to: {out_path.resolve()}")
    else:
        draw_with_pil(image_path, boxes, class_names, out_path)


if __name__ == "__main__":
    main()
