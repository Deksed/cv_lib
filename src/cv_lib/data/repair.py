"""Auto-repair YOLO labels flagged by :mod:`cv_lib.data.inspect`.

``inspect_dataset`` *reports* problems; this *fixes* them: clip boxes that spill
outside the image, and drop lines that can't be salvaged (too few fields,
non-numeric, out-of-range class id, or degenerate after clipping). Writes the
cleaned labels to a new directory by default so the originals stay intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RepairReport:
    """Result of :func:`repair_labels`."""

    out_dir: Path
    files: int = 0
    boxes_kept: int = 0
    boxes_clipped: int = 0
    boxes_dropped: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def _bump(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def print(self) -> None:
        print(f"Repair -> {self.out_dir}  ({self.files} files)")
        print(f"  kept={self.boxes_kept}  clipped={self.boxes_clipped}  dropped={self.boxes_dropped}")
        for reason in sorted(self.reasons):
            print(f"    dropped [{reason}]: {self.reasons[reason]}")


def _clip_box(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    """Clip a normalised box to the unit frame; return None if degenerate."""
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 0 or nh <= 0:
        return None
    return (x1 + nw / 2, y1 + nh / 2, nw, nh)


def repair_labels(
    labels_dir: str | Path,
    *,
    num_classes: int | None = None,
    clip: bool = True,
    out_dir: str | Path | None = None,
) -> RepairReport:
    """Clean a directory of YOLO labels in place or into ``out_dir``.

    Args:
        labels_dir: Directory of YOLO ``.txt`` label files.
        num_classes: If given, drop boxes whose class id is outside
            ``[0, num_classes)``.
        clip: Clip boxes that extend past the image edge back into ``[0, 1]``
            instead of dropping them (degenerate results are still dropped).
        out_dir: Where to write cleaned labels. ``None`` overwrites in place.

    Returns:
        A :class:`RepairReport` summarising kept / clipped / dropped boxes.
    """
    labels_dir = Path(labels_dir)
    dst_dir = Path(out_dir) if out_dir is not None else labels_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    report = RepairReport(out_dir=dst_dir)

    for label_file in sorted(labels_dir.glob("*.txt")):
        report.files += 1
        out_lines: list[str] = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                report.boxes_dropped += 1
                report._bump("too few fields")
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                report.boxes_dropped += 1
                report._bump("non-numeric")
                continue
            extra = parts[5:]  # preserve trailing columns (e.g. confidence)

            if num_classes is not None and not (0 <= cid < num_classes):
                report.boxes_dropped += 1
                report._bump("class id out of range")
                continue

            # In bounds means the whole box (not just its centre) sits inside the
            # unit frame — a box spilling past an edge is clipped back in.
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            in_bounds = w > 0 and h > 0 and x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1
            if not in_bounds:
                if not clip:
                    report.boxes_dropped += 1
                    report._bump("out of bounds")
                    continue
                clipped = _clip_box(cx, cy, w, h)
                if clipped is None:
                    report.boxes_dropped += 1
                    report._bump("degenerate after clip")
                    continue
                cx, cy, w, h = clipped
                report.boxes_clipped += 1

            report.boxes_kept += 1
            coords = f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            out_lines.append(" ".join([str(cid), coords, *extra]).rstrip())

        (dst_dir / label_file.name).write_text("\n".join(out_lines))

    return report
