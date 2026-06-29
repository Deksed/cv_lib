"""Class-id remapping / filtering for YOLO label sets.

Merge, rename or drop classes across a directory of YOLO ``.txt`` labels and
(optionally) rewrite the matching ``data.yaml``. Common when merging datasets
with different taxonomies or collapsing fine-grained classes into coarse ones.

Each label line is ``<class_id> <cx> <cy> <w> <h> [conf]``; only the leading
class id is touched, the geometry (and any trailing confidence column) is kept
verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RemapReport:
    """Result of :func:`remap_labels`."""

    out_dir: Path
    files: int
    boxes_kept: int
    boxes_dropped: int
    boxes_remapped: int
    new_class_counts: dict[int, int] = field(default_factory=dict)
    data_yaml: Path | None = None

    def print(self) -> None:
        print(f"Remap -> {self.out_dir}  ({self.files} files)")
        print(f"  kept={self.boxes_kept}  remapped={self.boxes_remapped}  dropped={self.boxes_dropped}")
        for cid in sorted(self.new_class_counts):
            print(f"    class {cid}: {self.new_class_counts[cid]}")
        if self.data_yaml is not None:
            print(f"  data.yaml: {self.data_yaml}")


def remap_labels(
    labels_dir: str | Path,
    mapping: dict[int, int] | None = None,
    *,
    drop: set[int] | list[int] | None = None,
    out_dir: str | Path | None = None,
    class_names: list[str] | None = None,
    data_yaml: str | Path | None = None,
) -> RemapReport:
    """Remap and/or drop class ids across a directory of YOLO labels.

    Args:
        labels_dir: Directory of YOLO ``.txt`` label files.
        mapping: ``{old_id: new_id}``. Ids absent from the mapping are kept
            unchanged (unless dropped). Use it to merge (several old → one new)
            or renumber classes.
        drop: Class ids to remove entirely (applied *before* remapping, i.e.
            these refer to the original ids).
        out_dir: Where to write the rewritten labels. ``None`` overwrites the
            files in place.
        class_names: New ordered class names to write into ``data.yaml`` (only
            used when ``data_yaml`` is given).
        data_yaml: Path to a ``data.yaml`` to rewrite with ``class_names`` /
            updated ``nc``. Skipped when ``None``.

    Returns:
        A :class:`RemapReport` with per-class counts after remapping.
    """
    mapping = {int(k): int(v) for k, v in (mapping or {}).items()}
    drop_set = {int(c) for c in (drop or set())}

    labels_dir = Path(labels_dir)
    dst_dir = Path(out_dir) if out_dir is not None else labels_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    kept = dropped = remapped = 0
    counts: dict[int, int] = {}

    for label_file in sorted(labels_dir.glob("*.txt")):
        out_lines: list[str] = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                old_id = int(float(parts[0]))
            except ValueError:
                continue
            if old_id in drop_set:
                dropped += 1
                continue
            new_id = mapping.get(old_id, old_id)
            if new_id != old_id:
                remapped += 1
            kept += 1
            counts[new_id] = counts.get(new_id, 0) + 1
            out_lines.append(" ".join([str(new_id), *parts[1:]]))
        (dst_dir / label_file.name).write_text("\n".join(out_lines))

    report = RemapReport(
        out_dir=dst_dir,
        files=len(list(dst_dir.glob("*.txt"))),
        boxes_kept=kept,
        boxes_dropped=dropped,
        boxes_remapped=remapped,
        new_class_counts=counts,
    )

    if data_yaml is not None and class_names is not None:
        report.data_yaml = _rewrite_data_yaml(Path(data_yaml), class_names, out_dir)

    return report


def _rewrite_data_yaml(
    src: Path, class_names: list[str], out_dir: str | Path | None
) -> Path:
    """Rewrite ``names``/``nc`` of a data.yaml; write next to ``out_dir`` if given."""
    doc = yaml.safe_load(src.read_text()) if src.exists() else {}
    doc["names"] = class_names
    doc["nc"] = len(class_names)
    dst = Path(out_dir) / "data.yaml" if out_dir is not None else src
    dst.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dst


def parse_mapping(pairs: list[str]) -> dict[int, int]:
    """Parse ``["2=0", "3=0"]`` CLI tokens into ``{2: 0, 3: 0}``."""
    mapping: dict[int, int] = {}
    for token in pairs:
        if "=" not in token:
            raise ValueError(f"Invalid --map token {token!r}; expected OLD=NEW.")
        old, new = token.split("=", 1)
        mapping[int(old)] = int(new)
    return mapping
