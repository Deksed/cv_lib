"""Merge several YOLO datasets into one with a unified class taxonomy.

Combining datasets that were labelled independently means their class ids rarely
line up. This builds a union of class *names* (first-seen order), remaps every
source's ids onto that shared index, copies images + labels into one dataset
(filenames prefixed per source to avoid collisions), and writes a merged
``data.yaml``. Pairs with :mod:`cv_lib.data.remap` (single-dataset renumbering).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cv_lib.data import class_names_from_yaml, iter_image_label_pairs

_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


@dataclass
class DatasetSource:
    """One dataset to merge: images, labels and its ordered class names."""

    images_dir: Path
    labels_dir: Path | None
    class_names: list[str]


@dataclass
class MergeReport:
    """Result of :func:`merge_datasets`."""

    out_dir: Path
    class_names: list[str]
    images: int = 0
    per_source: dict[str, int] = field(default_factory=dict)
    data_yaml: Path | None = None

    def print(self) -> None:
        print(f"Merge -> {self.out_dir}  ({self.images} images, nc={len(self.class_names)})")
        for name, n in self.per_source.items():
            print(f"  {name}: {n}")
        print(f"  classes: {self.class_names}")
        if self.data_yaml is not None:
            print(f"  data.yaml: {self.data_yaml}")


def source_from_root(root: str | Path, data_yaml: str = "data.yaml") -> DatasetSource:
    """Build a :class:`DatasetSource` from a dataset root.

    Expects ``<root>/images``, ``<root>/labels`` and ``<root>/<data_yaml>``.
    """
    root = Path(root)
    names_path = root / data_yaml
    names = class_names_from_yaml(names_path) if names_path.exists() else []
    return DatasetSource(images_dir=root / "images", labels_dir=root / "labels", class_names=names)


def _place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
    else:  # "copy"
        shutil.copy2(src, dst)


def merge_datasets(
    sources: list[DatasetSource],
    out_dir: str | Path,
    *,
    mode: str = "copy",
    write_yaml: bool = True,
    extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
) -> MergeReport:
    """Merge YOLO datasets into ``out_dir`` under a unified class taxonomy.

    Args:
        sources: Datasets to merge. Class ids are remapped to a union of class
            *names* (sources with empty ``class_names`` keep their numeric ids).
        out_dir: Destination root; gets ``images/`` + ``labels/`` + ``data.yaml``.
        mode: ``"copy"`` (default) or ``"move"``. Files are prefixed ``s<i>_``.
        write_yaml: Whether to emit ``<out_dir>/data.yaml``.
        extensions: Image extensions to include.

    Returns:
        A :class:`MergeReport`.
    """
    # Union of class names (first-seen order across sources).
    unified: list[str] = []
    for src in sources:
        for name in src.class_names:
            if name not in unified:
                unified.append(name)

    out_dir = Path(out_dir)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    report = MergeReport(out_dir=out_dir, class_names=unified)

    for i, src in enumerate(sources):
        # local class id -> unified id (identity when names are unknown)
        if src.class_names:
            remap = {local: unified.index(name) for local, name in enumerate(src.class_names)}
        else:
            remap = {}
        count = 0
        for img_path, label_path in iter_image_label_pairs(src.images_dir, src.labels_dir, extensions):
            stem = f"s{i}_{img_path.stem}"
            _place(img_path, images_out / f"{stem}{img_path.suffix}", mode)
            if label_path.exists():
                out_lines: list[str] = []
                for line in label_path.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    cid = int(float(parts[0]))
                    new_id = remap.get(cid, cid)
                    out_lines.append(" ".join([str(new_id), *parts[1:]]))
                (labels_out / f"{stem}.txt").parent.mkdir(parents=True, exist_ok=True)
                (labels_out / f"{stem}.txt").write_text("\n".join(out_lines))
            count += 1
        report.per_source[str(src.images_dir)] = count
        report.images += count

    if write_yaml and unified:
        doc = {
            "path": str(out_dir.resolve()),
            "train": "images",
            "val": "images",
            "nc": len(unified),
            "names": unified,
        }
        data_yaml = out_dir / "data.yaml"
        out_dir.mkdir(parents=True, exist_ok=True)
        data_yaml.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
        report.data_yaml = data_yaml

    return report
