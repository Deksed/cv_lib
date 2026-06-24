#!/usr/bin/env python
"""
Compare ground truth annotations vs model predictions for a single image.

Usage examples:

  # Basic — auto-resolves label next to image, displays result:
  python scripts/compare_gt_pred.py image.jpg --model runs/train/best.pt --names car person

  # Save instead of display:
  python scripts/compare_gt_pred.py image.jpg --model best.pt --names car person --output out.jpg --no-show

  # Explicit label path:
  python scripts/compare_gt_pred.py image.jpg --model best.pt --names car person --label labels/image.txt

  # Lower confidence threshold:
  python scripts/compare_gt_pred.py image.jpg --model best.pt --names car person --conf 0.15
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_lib.viz.compare import compare_gt_pred


def _load_names_from_yaml(yaml_path: str) -> list[str]:
    """Parse class names from a YOLO data.yaml."""
    import yaml
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="GT vs prediction side-by-side visualizer.")

    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--model", required=True, help="Path to Ultralytics .pt model.")

    names_group = parser.add_mutually_exclusive_group(required=True)
    names_group.add_argument("--names", nargs="+", metavar="NAME", help="Class names in order, e.g. --names car person.")
    names_group.add_argument("--data", metavar="YAML", help="Path to YOLO data.yaml (reads names from it).")

    parser.add_argument("--label", default=None, help="Explicit path to YOLO .txt label (auto-resolved if omitted).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions (default: 0.25).")
    parser.add_argument("--output", default=None, help="Save result to this path instead of (or in addition to) display.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a display window (useful on headless servers).")

    args = parser.parse_args()

    # DATA_ROOT applied to relative paths if set
    data_root = os.environ.get("DATA_ROOT", "")
    image_path = Path(args.image)
    if not image_path.is_absolute() and data_root:
        image_path = Path(data_root) / image_path

    class_names = _load_names_from_yaml(args.data) if args.data else args.names

    compare_gt_pred(
        image_path=image_path,
        model_path=args.model,
        class_names=class_names,
        conf_threshold=args.conf,
        label_path=args.label,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
