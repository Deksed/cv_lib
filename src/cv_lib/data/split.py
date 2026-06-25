"""Train/val/test split for YOLO datasets + ``data.yaml`` generation.

Splits an ``(images_dir, labels_dir)`` pair into train/val/test subsets, places
files into the YOLO-standard layout (``<out>/images/<split>`` +
``<out>/labels/<split>``), and writes a ``data.yaml``. With
``stratify_by_class`` each image is bucketed by its *dominant* class (the most
frequent class id in its label) so rare classes stay represented in every split.

This is also the ``split`` stage of the DVC pipeline (see
:mod:`cv_lib.data.dvc_gen`). Runnable as a module: ``python -m cv_lib.data.split``
(forwards to ``cvlib split``).
"""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from cv_lib.data import iter_image_label_pairs

# Canonical split names, paired positionally with the given ratios.
SPLITS: tuple[str, ...] = ("train", "val", "test")
_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
_BACKGROUND = -1  # stratum for images with no labels


@dataclass
class SplitReport:
    """Result of :func:`train_val_test_split`."""

    out_dir: Path
    counts: dict[str, int]
    num_classes: int
    data_yaml: Path | None

    def print(self) -> None:
        print(f"Split -> {self.out_dir}  (nc={self.num_classes}, total={sum(self.counts.values())})")
        for name, n in self.counts.items():
            print(f"  {name:5s}: {n}")
        if self.data_yaml is not None:
            print(f"  data.yaml: {self.data_yaml}")


def _label_classes(label_path: Path) -> list[int]:
    """Return the class ids in a YOLO label file (empty if missing/blank)."""
    if not label_path.exists():
        return []
    classes: list[int] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            try:
                classes.append(int(float(parts[0])))
            except ValueError:
                continue
    return classes


def _partition(items: list, ratios: tuple[float, ...], names: tuple[str, ...]) -> dict[str, list]:
    """Slice ``items`` into named chunks by ``ratios`` (remainder → earliest splits)."""
    n = len(items)
    sizes = [int(r * n) for r in ratios]
    for i in range(n - sum(sizes)):  # hand out the rounding remainder
        sizes[i % len(sizes)] += 1
    out: dict[str, list] = {}
    start = 0
    for name, size in zip(names, sizes):
        out[name] = items[start : start + size]
        start += size
    return out


def _place(src: Path, dst_dir: Path, mode: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
        except OSError:  # Windows without privilege / cross-device → fall back
            shutil.copy2(src, dst)
    else:  # pragma: no cover - guarded by argparse/choices
        raise ValueError(f"Unknown mode: {mode!r}")


def _write_data_yaml(
    out_dir: Path, split_names: tuple[str, ...], num_classes: int, names: list[str]
) -> Path:
    doc: dict = {"path": str(out_dir.resolve())}
    for split in split_names:
        doc[split] = f"images/{split}"
    doc["nc"] = num_classes
    doc["names"] = names
    path = out_dir / "data.yaml"
    path.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def train_val_test_split(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    out_dir: str | Path = "dataset",
    *,
    ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    seed: int = 42,
    stratify_by_class: bool = True,
    class_names: list[str] | None = None,
    mode: str = "copy",
    write_yaml: bool = True,
    extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
) -> SplitReport:
    """Split a YOLO dataset into train/val/test and lay it out under ``out_dir``.

    Args:
        images_dir: Directory of images.
        labels_dir: Directory of YOLO ``.txt`` labels (inferred from
            ``images_dir`` if ``None`` — see :func:`iter_image_label_pairs`).
        out_dir: Destination root; gets ``images/<split>`` + ``labels/<split>``.
        ratios: Two or three fractions summing to 1.0. Two values produce only
            train/val (no test split).
        seed: RNG seed for the shuffle (deterministic output).
        stratify_by_class: Bucket each image by its dominant class before
            splitting so the class mix is preserved across splits.
        class_names: Class names for ``data.yaml``; inferred as ``["0", "1", ...]``
            from the labels when omitted.
        mode: ``"copy"`` (default), ``"symlink"`` (copy fallback on Windows), or
            ``"move"``.
        write_yaml: Whether to emit ``<out_dir>/data.yaml``.
        extensions: Image extensions to include.

    Returns:
        A :class:`SplitReport` with per-split counts and the ``data.yaml`` path.
    """
    if len(ratios) not in (2, 3):
        raise ValueError(f"ratios must have 2 or 3 values, got {len(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    pairs = iter_image_label_pairs(images_dir, labels_dir, extensions)
    if not pairs:
        raise ValueError(f"No images with extensions {extensions} in {images_dir}")

    split_names = SPLITS[: len(ratios)]

    # Bucket pairs by stratum (dominant class, or a single bucket if disabled).
    groups: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    max_class = -1
    for img, lbl in pairs:
        classes = _label_classes(lbl)
        if classes:
            max_class = max(max_class, *classes)
        stratum = Counter(classes).most_common(1)[0][0] if classes else _BACKGROUND
        groups[0 if not stratify_by_class else stratum].append((img, lbl))

    import random

    rng = random.Random(seed)
    assigned: dict[str, list[tuple[Path, Path]]] = {name: [] for name in split_names}
    for stratum in sorted(groups):  # sort strata for reproducibility
        bucket = groups[stratum]
        rng.shuffle(bucket)
        for name, chunk in _partition(bucket, ratios, split_names).items():
            assigned[name].extend(chunk)

    for split, items in assigned.items():
        for img, lbl in items:
            _place(img, Path(out_dir) / "images" / split, mode)
            if lbl.exists():
                _place(lbl, Path(out_dir) / "labels" / split, mode)

    num_classes = len(class_names) if class_names is not None else max_class + 1
    names = class_names if class_names is not None else [str(i) for i in range(num_classes)]

    data_yaml = None
    if write_yaml:
        data_yaml = _write_data_yaml(Path(out_dir), split_names, num_classes, names)

    return SplitReport(
        out_dir=Path(out_dir),
        counts={name: len(items) for name, items in assigned.items()},
        num_classes=num_classes,
        data_yaml=data_yaml,
    )


def main(argv: list[str] | None = None) -> int:
    """Module entry point — forwards to ``cvlib split`` so the DVC stage works."""
    import sys

    from cv_lib.cli import main as cli_main

    return cli_main(["split", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
