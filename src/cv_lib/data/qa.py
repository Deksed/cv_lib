"""Annotation QA — surface *suspicious but valid* YOLO boxes.

Complements :mod:`cv_lib.data.inspect` (which catches broken / out-of-bounds /
missing labels). This audit flags annotations that parse fine yet look wrong in
practice: micro and full-frame boxes, extreme aspect ratios, exact duplicate
boxes, and images whose object count is a statistical outlier. The goal is a
fast triage list before training, not a hard validation gate.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Finding:
    """A single flagged annotation (or whole-file) issue."""

    file: str
    kind: str  # "tiny" | "huge" | "aspect" | "duplicate" | "count_outlier"
    detail: str


@dataclass
class QAReport:
    """Result of :func:`audit_labels`."""

    findings: list[Finding] = field(default_factory=list)
    files_checked: int = 0
    boxes_checked: int = 0

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        return counts

    def print(self, limit: int = 20) -> None:
        counts = self.by_kind()
        print(f"QA: {self.files_checked} files, {self.boxes_checked} boxes, {len(self.findings)} findings")
        for kind in sorted(counts):
            print(f"  {kind:14s}: {counts[kind]}")
        if self.findings:
            print("  --- examples ---")
            for f in self.findings[:limit]:
                print(f"  [{f.kind}] {f.file}: {f.detail}")
            if len(self.findings) > limit:
                print(f"  … and {len(self.findings) - limit} more")


def _iter_boxes(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append((cid, cx, cy, w, h))
    return boxes


def audit_labels(
    labels_dir: str | Path,
    *,
    min_box_area: float = 0.0005,
    max_box_area: float = 0.9,
    max_aspect: float = 10.0,
    count_z: float = 3.0,
) -> QAReport:
    """Audit a directory of YOLO labels for suspicious annotations.

    Args:
        labels_dir: Directory of YOLO ``.txt`` label files.
        min_box_area: Boxes with normalised area (``w*h``) below this are
            flagged ``tiny`` (default ~0.05% of the frame).
        max_box_area: Boxes with area above this are flagged ``huge``.
        max_aspect: Boxes whose longer/shorter side ratio exceeds this are
            flagged ``aspect``.
        count_z: Files whose object count is more than this many standard
            deviations above the mean are flagged ``count_outlier``.

    Returns:
        A :class:`QAReport` listing every :class:`Finding`.
    """
    report = QAReport()
    per_file_counts: dict[str, int] = {}

    for label_file in sorted(Path(labels_dir).glob("*.txt")):
        report.files_checked += 1
        boxes = _iter_boxes(label_file)
        per_file_counts[label_file.name] = len(boxes)
        seen: set[tuple] = set()
        for cid, cx, cy, w, h in boxes:
            report.boxes_checked += 1
            area = w * h
            if area < min_box_area:
                report.findings.append(Finding(label_file.name, "tiny", f"class {cid} area={area:.5f}"))
            elif area > max_box_area:
                report.findings.append(Finding(label_file.name, "huge", f"class {cid} area={area:.3f}"))
            if w > 0 and h > 0:
                aspect = max(w / h, h / w)
                if aspect > max_aspect:
                    report.findings.append(
                        Finding(label_file.name, "aspect", f"class {cid} aspect={aspect:.1f}")
                    )
            key = (cid, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
            if key in seen:
                report.findings.append(Finding(label_file.name, "duplicate", f"class {cid} at ({cx:.3f},{cy:.3f})"))
            seen.add(key)

    # Per-image object-count outliers (needs >=2 files with boxes to be meaningful).
    counts = [c for c in per_file_counts.values() if c > 0]
    if len(counts) >= 2:
        mean = statistics.mean(counts)
        std = statistics.pstdev(counts)
        if std > 0:
            threshold = mean + count_z * std
            for name, c in per_file_counts.items():
                if c > threshold:
                    report.findings.append(
                        Finding(name, "count_outlier", f"{c} objects (mean={mean:.1f}, std={std:.1f})")
                    )

    return report
