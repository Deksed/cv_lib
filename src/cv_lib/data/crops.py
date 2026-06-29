"""Extract object crops from a YOLO dataset.

Cuts every annotated box out of its image and writes the crops to disk, by
default grouped into ``<out>/<class>/`` folders. Handy for eyeballing label
quality class-by-class or for bootstrapping a classification dataset from
detection labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from cv_lib.data import iter_image_label_pairs

_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


@dataclass
class CropReport:
    """Result of :func:`extract_crops`."""

    out_dir: Path
    crops: int
    images: int
    per_class: dict[str, int] = field(default_factory=dict)

    def print(self) -> None:
        print(f"Crops -> {self.out_dir}  ({self.crops} crops from {self.images} images)")
        for name in sorted(self.per_class):
            print(f"  {name}: {self.per_class[name]}")


def _yolo_to_pixels(cx: float, cy: float, w: float, h: float, iw: int, ih: int, pad: float):
    """Convert a normalised YOLO box to clipped pixel corners with fractional pad."""
    bw, bh = w * iw, h * ih
    px, py = cx * iw, cy * ih
    half_w = bw / 2 * (1 + pad)
    half_h = bh / 2 * (1 + pad)
    x1 = max(0, int(round(px - half_w)))
    y1 = max(0, int(round(py - half_h)))
    x2 = min(iw, int(round(px + half_w)))
    y2 = min(ih, int(round(py + half_h)))
    return x1, y1, x2, y2


def extract_crops(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    out_dir: str | Path = "crops",
    *,
    per_class: bool = True,
    pad: float = 0.0,
    class_names: list[str] | None = None,
    extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
) -> CropReport:
    """Crop every labelled object out of a YOLO dataset onto disk.

    Args:
        images_dir: Directory of images.
        labels_dir: Directory of YOLO ``.txt`` labels (inferred from
            ``images_dir`` if ``None``).
        out_dir: Output root. With ``per_class`` each crop lands in
            ``<out_dir>/<class>/``; otherwise flat in ``<out_dir>/``.
        per_class: Group crops into per-class subfolders.
        pad: Fractional padding added around each box before cropping
            (``0.1`` = 10% larger), clipped to image bounds.
        class_names: Names used for the per-class folders; falls back to the
            numeric class id when omitted or out of range.
        extensions: Image extensions to include.

    Returns:
        A :class:`CropReport` with total and per-class crop counts.
    """
    out_dir = Path(out_dir)
    report = CropReport(out_dir=out_dir, crops=0, images=0)

    for img_path, label_path in iter_image_label_pairs(images_dir, labels_dir, extensions):
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        report.images += 1
        idx = 0
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            x1, y1, x2, y2 = _yolo_to_pixels(cx, cy, w, h, iw, ih, pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            name = (
                class_names[cid] if class_names and 0 <= cid < len(class_names) else str(cid)
            )
            dst_dir = out_dir / name if per_class else out_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst_dir / f"{img_path.stem}_{idx}.jpg"), crop)
            report.crops += 1
            report.per_class[name] = report.per_class.get(name, 0) + 1
            idx += 1

    return report
