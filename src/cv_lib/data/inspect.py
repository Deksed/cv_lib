"""Dataset health checks: missing labels, corrupt images, bbox sanity, class imbalance."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class InspectReport:
    images_total: int = 0
    corrupt_images: list[Path] = field(default_factory=list)
    missing_labels: list[Path] = field(default_factory=list)
    empty_labels: list[Path] = field(default_factory=list)
    invalid_boxes: list[tuple[Path, int, str]] = field(default_factory=list)
    class_counts: np.ndarray | None = None
    class_names: list[str] = field(default_factory=list)

    def print(self) -> None:
        print("\nDataset inspection report")
        print("=" * 50)
        print(f"  Total images      : {self.images_total}")
        print(f"  Corrupt images    : {len(self.corrupt_images)}")
        print(f"  Missing labels    : {len(self.missing_labels)}")
        print(f"  Empty label files : {len(self.empty_labels)}")
        print(f"  Invalid boxes     : {len(self.invalid_boxes)}")

        if self.corrupt_images:
            print("\nCorrupt images:")
            for p in self.corrupt_images:
                print(f"  {p}")

        if self.missing_labels:
            print("\nMissing label files:")
            for p in self.missing_labels[:20]:
                print(f"  {p}")
            if len(self.missing_labels) > 20:
                print(f"  … and {len(self.missing_labels) - 20} more")

        if self.invalid_boxes:
            print("\nInvalid boxes (file, line, reason):")
            for p, line_no, reason in self.invalid_boxes[:20]:
                print(f"  {p}:{line_no}  {reason}")
            if len(self.invalid_boxes) > 20:
                print(f"  … and {len(self.invalid_boxes) - 20} more")

        if self.class_counts is not None and len(self.class_counts):
            print("\nClass distribution:")
            names = self.class_names or [str(i) for i in range(len(self.class_counts))]
            max_name = max(len(n) for n in names)
            total = self.class_counts.sum()
            for name, count in zip(names, self.class_counts):
                bar = "█" * int(40 * count / max(total, 1))
                print(f"  {name:<{max_name}}  {count:>7}  {bar}")
        print()


def inspect_dataset(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    num_classes: int | None = None,
    class_names: list[str] | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> InspectReport:
    """
    Run health checks on a YOLO dataset split.

    Args:
        images_dir:   path to images directory
        labels_dir:   path to labels directory; inferred from images_dir if None
        num_classes:  expected number of classes (for OOB class id check)
        class_names:  class name list for the report
        extensions:   image file extensions to scan

    Returns:
        InspectReport with all findings populated.
    """
    import cv2

    images_dir = Path(images_dir)
    if labels_dir is None:
        parts = images_dir.parts
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            labels_dir = Path(*parts[:idx], "labels", *parts[idx + 1:])
        else:
            labels_dir = images_dir
    labels_dir = Path(labels_dir)

    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in extensions)
    report = InspectReport(images_total=len(image_files), class_names=class_names or [])

    n_classes = num_classes or (len(class_names) if class_names else 0)
    class_counts = np.zeros(max(n_classes, 1), dtype=np.int64)

    for img_path in image_files:
        # --- corrupt image check ---
        img = cv2.imread(str(img_path))
        if img is None:
            report.corrupt_images.append(img_path)
            continue

        img_h, img_w = img.shape[:2]

        label_path = (labels_dir / img_path.stem).with_suffix(".txt")
        if not label_path.exists():
            report.missing_labels.append(img_path)
            continue

        text = label_path.read_text().strip()
        if not text:
            report.empty_labels.append(label_path)
            continue

        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts_line = line.split()
            if len(parts_line) < 5:
                report.invalid_boxes.append((label_path, line_no, "fewer than 5 fields"))
                continue

            try:
                cid = int(parts_line[0])
                cx, cy, bw, bh = map(float, parts_line[1:5])
            except ValueError:
                report.invalid_boxes.append((label_path, line_no, "non-numeric fields"))
                continue

            # validate class id
            if n_classes and not (0 <= cid < n_classes):
                report.invalid_boxes.append(
                    (label_path, line_no, f"class id {cid} out of range [0, {n_classes})")
                )
                continue

            # validate bbox coords
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                report.invalid_boxes.append(
                    (label_path, line_no, f"center out of [0,1]: cx={cx:.4f} cy={cy:.4f}")
                )
                continue
            if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                report.invalid_boxes.append(
                    (label_path, line_no, f"size out of (0,1]: w={bw:.4f} h={bh:.4f}")
                )
                continue

            # only count fully valid boxes
            if 0 <= cid < len(class_counts):
                class_counts[cid] += 1
            elif cid >= len(class_counts):
                class_counts = np.append(class_counts, np.zeros(cid - len(class_counts) + 1, dtype=np.int64))
                class_counts[cid] += 1

    report.class_counts = class_counts[:n_classes] if n_classes else class_counts
    return report
