"""Train/val/test splits computed directly on the flat CVAT CSV.

The CSV is the source of truth (one row per instance; group by image), so these
splitters read it, assign each *image* to a split, and write per-split CSVs plus a
manifest — they never copy image/label files (that stays with
:func:`cv_lib.data.split.train_val_test_split`).

Three strategies, all the *same* greedy stratified engine with a different
grouping function — the grouping is the only thing that changes:

* :func:`random_split_csv` — every image is its own group (plain stratified split).
* :func:`temporal_split_csv` — frames close in ``ts`` share a group, so
  near-duplicate consecutive frames land in the *same* split (no leakage).
* :func:`camera_temporal_split_csv` — the same time-based grouping done *per
  camera*, so identical timestamps on different cameras aren't wrongly merged.

Stratification: whole groups move together, so a group's stratum is the most
common per-image dominant class within it, and groups are handed to whichever
split is currently furthest below its target ratio (the standard grouped-
stratified greedy — no pandas/sklearn needed).
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from cv_lib.data.convert import _read_cvat_csv, _write_cvat_csv

SPLITS: tuple[str, ...] = ("train", "val", "test")
_BACKGROUND = "__background__"  # stratum for images with no labels
# Label header is misspelled "instance_lable" upstream; tolerate both.
_LABEL_COLUMNS: tuple[str, ...] = ("instance_label", "instance_lable")


@dataclass
class ImageRecord:
    """Per-image view of the CSV rows (one image, many instance rows)."""

    name: str
    labels: Counter[str] = field(default_factory=Counter)
    ts: float | None = None
    camera: str | None = None

    @property
    def stratum(self) -> str:
        """Dominant class label, or the background sentinel when unlabeled."""
        return self.labels.most_common(1)[0][0] if self.labels else _BACKGROUND


@dataclass
class CsvSplitReport:
    """Result of a CSV splitter."""

    assignment: dict[str, str]  # image_name -> split
    counts: dict[str, int]  # split -> image count
    out_dir: Path | None = None
    files: dict[str, Path] = field(default_factory=dict)  # split/"manifest" -> csv

    def print(self) -> None:
        total = sum(self.counts.values())
        print(f"CSV split  (images={total})")
        for name, n in self.counts.items():
            print(f"  {name:5s}: {n}")
        if self.out_dir is not None:
            print(f"  out: {self.out_dir}")


# ---------------------------------------------------------------------------
# Reading / parsing
# ---------------------------------------------------------------------------

def _validate_ratios(ratios: tuple[float, ...]) -> None:
    if len(ratios) not in (2, 3):
        raise ValueError(f"ratios must have 2 or 3 values, got {len(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")


def _get_label(row: dict[str, str], label_column: str | None) -> str:
    if label_column is not None:
        return row.get(label_column, "")
    for col in _LABEL_COLUMNS:
        if row.get(col):
            return row[col]
    return ""


def _parse_ts(value: str) -> float | None:
    """Parse a timestamp cell to float seconds; None if empty/unparseable.

    Accepts a numeric epoch (kept as-is, so ``gap`` is in the same unit) or an
    ISO-8601 datetime string (converted to epoch seconds).
    """
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    from datetime import datetime

    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _read_image_records(
    csv_path: str | Path,
    *,
    label_column: str | None,
    ts_column: str | None,
    camera_column: str | None,
) -> tuple[list[ImageRecord], list[str], list[dict[str, str]]]:
    """Collapse CSV rows into per-image :class:`ImageRecord` objects.

    Returns ``(records, header, rows)`` so callers can also re-emit the raw rows.
    """
    header, rows = _read_cvat_csv(csv_path)
    for col in (ts_column, camera_column):
        if col is not None and col not in header:
            raise ValueError(f"Column {col!r} not in CSV header; available: {header}")

    records: dict[str, ImageRecord] = {}
    for row in rows:
        raw = row.get("image_name") or row.get("image_path")
        if not raw:
            continue
        # Key by basename: image names are `camera_ts.<ext>` (camera encoded in
        # the filename), so the basename is unique per camera and safe to group
        # on. If a source ever reuses a bare filename across cameras, switch this
        # to the `image_id` column.
        name = Path(raw).name
        rec = records.get(name)
        if rec is None:
            rec = ImageRecord(name=name)
            if ts_column is not None:
                rec.ts = _parse_ts(row.get(ts_column, ""))
            if camera_column is not None:
                rec.camera = (row.get(camera_column) or "").strip()
            records[name] = rec
        label = _get_label(row, label_column)
        if label:
            rec.labels[label] += 1

    return list(records.values()), header, rows


# ---------------------------------------------------------------------------
# Grouping functions: list[ImageRecord] -> list[list[str]] (groups of image names)
# ---------------------------------------------------------------------------

def _groups_per_image(records: list[ImageRecord]) -> list[list[str]]:
    return [[r.name] for r in records]


def _sessionize(records: list[ImageRecord], gap: float) -> list[list[str]]:
    """Group frames by ts proximity; a gap > ``gap`` starts a new group.

    Records without a ts become singleton groups (can't be time-grouped).
    """
    timed = sorted((r for r in records if r.ts is not None), key=lambda r: r.ts)
    groups: list[list[str]] = []
    current: list[str] = []
    prev: float | None = None
    for r in timed:
        if prev is not None and (r.ts - prev) > gap:
            groups.append(current)
            current = []
        current.append(r.name)
        prev = r.ts
    if current:
        groups.append(current)
    groups.extend([r.name] for r in records if r.ts is None)
    return groups


def _warn_missing_ts(records: list[ImageRecord], ts_column: str) -> None:
    """Warn when frames lack a parseable ``ts`` — they become singleton groups,
    so a temporal split silently degrades toward a random one."""
    missing = sum(1 for r in records if r.ts is None)
    if not missing:
        return
    if missing == len(records):
        logger.warning(
            "No parseable {!r} on any of {} image(s); temporal grouping is a no-op "
            "(each frame is its own group — effectively a random split).",
            ts_column, len(records),
        )
    elif missing / len(records) > 0.5:
        logger.warning(
            "{}/{} image(s) lack a parseable {!r}; those become singleton groups.",
            missing, len(records), ts_column,
        )


def _groups_camera_temporal(records: list[ImageRecord], gap: float) -> list[list[str]]:
    by_camera: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        by_camera[r.camera or ""].append(r)
    groups: list[list[str]] = []
    for camera in sorted(by_camera):  # deterministic camera order
        groups.extend(_sessionize(by_camera[camera], gap))
    return groups


# ---------------------------------------------------------------------------
# Stratified greedy assignment of whole groups
# ---------------------------------------------------------------------------

def _assign_groups(
    groups: list[list[str]],
    stratum_of: dict[str, str],
    ratios: tuple[float, ...],
    names: tuple[str, ...],
    *,
    seed: int,
    stratify: bool,
) -> dict[str, str]:
    """Assign whole groups to splits, returning image_name -> split.

    Within each stratum, groups are shuffled then placed largest-first into
    whichever split is furthest below its target share — keeps ratios close
    while never splitting a group across two sets.
    """
    import random

    rng = random.Random(seed)

    buckets: dict[str, list[list[str]]] = defaultdict(list)
    for group in groups:
        if stratify:
            key = Counter(stratum_of[n] for n in group).most_common(1)[0][0]
        else:
            key = _BACKGROUND
        buckets[key].append(group)

    assignment: dict[str, str] = {}
    for stratum in sorted(buckets):
        group_list = buckets[stratum]
        rng.shuffle(group_list)
        group_list.sort(key=len, reverse=True)
        size = sum(len(g) for g in group_list)
        targets = {name: r * size for name, r in zip(names, ratios)}
        filled = {name: 0 for name in names}
        for group in group_list:
            # Largest deficit wins; ties break toward the earliest split name.
            pick = max(names, key=lambda n: (targets[n] - filled[n], -names.index(n)))
            for name in group:
                assignment[name] = pick
            filled[pick] += len(group)
    return assignment


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_outputs(
    header: list[str],
    rows: list[dict[str, str]],
    assignment: dict[str, str],
    names: tuple[str, ...],
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write per-split CSVs (original rows) + a split_manifest.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict[str, str]]] = {name: [] for name in names}
    for row in rows:
        raw = row.get("image_name") or row.get("image_path")
        if not raw:
            continue
        split = assignment.get(Path(raw).name)
        if split is not None:
            rows_by_split[split].append(row)

    files: dict[str, Path] = {}
    for name in names:
        path = out / f"{name}.csv"
        _write_cvat_csv(rows_by_split[name], path, header)
        files[name] = path

    manifest = out / "split_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "split"])
        for image_name, split in sorted(assignment.items()):
            writer.writerow([image_name, split])
    files["manifest"] = manifest
    return files


def _finalize(
    records: list[ImageRecord],
    groups: list[list[str]],
    header: list[str],
    rows: list[dict[str, str]],
    *,
    ratios: tuple[float, ...],
    seed: int,
    stratify: bool,
    out_dir: str | Path | None,
) -> CsvSplitReport:
    names = SPLITS[: len(ratios)]
    stratum_of = {r.name: r.stratum for r in records}
    assignment = _assign_groups(
        groups, stratum_of, ratios, names, seed=seed, stratify=stratify
    )
    counts = {name: 0 for name in names}
    for split in assignment.values():
        counts[split] += 1

    files: dict[str, Path] = {}
    if out_dir is not None:
        files = _write_outputs(header, rows, assignment, names, out_dir)

    return CsvSplitReport(
        assignment=assignment,
        counts=counts,
        out_dir=Path(out_dir) if out_dir is not None else None,
        files=files,
    )


# ---------------------------------------------------------------------------
# Public splitters
# ---------------------------------------------------------------------------

def random_split_csv(
    csv_path: str | Path,
    *,
    ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    seed: int = 42,
    stratify: bool = True,
    label_column: str | None = None,
    out_dir: str | Path | None = None,
) -> CsvSplitReport:
    """Plain stratified train/val/test split over the CSV's images.

    Each image is assigned independently, stratified by its dominant class. Use
    this when frames are already independent (no temporal correlation).

    Args:
        csv_path: Path to the flat CVAT CSV export.
        ratios: Two or three fractions summing to 1.0 (two → train/val only).
        seed: RNG seed for reproducible assignment.
        stratify: Balance each split by dominant class (off → ignore classes).
        label_column: Label column name; auto-detects ``instance_label`` /
            ``instance_lable`` when ``None``.
        out_dir: If given, write ``<split>.csv`` + ``split_manifest.csv`` here.

    Returns:
        A :class:`CsvSplitReport` (``assignment`` maps image_name → split).
    """
    _validate_ratios(ratios)
    records, header, rows = _read_image_records(
        csv_path, label_column=label_column, ts_column=None, camera_column=None
    )
    if not records:
        raise ValueError(f"No images found in {csv_path}")
    groups = _groups_per_image(records)
    return _finalize(
        records, groups, header, rows,
        ratios=ratios, seed=seed, stratify=stratify, out_dir=out_dir,
    )


def temporal_split_csv(
    csv_path: str | Path,
    *,
    ts_column: str = "ts",
    gap: float = 1.0,
    ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    seed: int = 42,
    stratify: bool = True,
    label_column: str | None = None,
    out_dir: str | Path | None = None,
) -> CsvSplitReport:
    """Time-grouped stratified split: frames close in ``ts`` share a split.

    Frames are sorted by ``ts`` and cut into sessions wherever the gap between
    consecutive frames exceeds ``gap``; a whole session goes to one split, so
    near-duplicate adjacent frames never straddle train/val (avoids leakage).

    Args:
        csv_path: Path to the flat CVAT CSV export.
        ts_column: Timestamp column (numeric epoch or ISO-8601 string).
        gap: Max spacing (in the ts unit) for two frames to stay in one session.
        ratios: Two or three fractions summing to 1.0.
        seed: RNG seed for reproducible assignment.
        stratify: Balance splits by each session's dominant class.
        label_column: Label column name; auto-detected when ``None``.
        out_dir: If given, write ``<split>.csv`` + ``split_manifest.csv`` here.

    Returns:
        A :class:`CsvSplitReport` (``assignment`` maps image_name → split).
    """
    _validate_ratios(ratios)
    records, header, rows = _read_image_records(
        csv_path, label_column=label_column, ts_column=ts_column, camera_column=None
    )
    if not records:
        raise ValueError(f"No images found in {csv_path}")
    _warn_missing_ts(records, ts_column)
    groups = _sessionize(records, gap)
    return _finalize(
        records, groups, header, rows,
        ratios=ratios, seed=seed, stratify=stratify, out_dir=out_dir,
    )


def camera_temporal_split_csv(
    csv_path: str | Path,
    *,
    camera_column: str = "camera",
    ts_column: str = "ts",
    gap: float = 1.0,
    ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    seed: int = 42,
    stratify: bool = True,
    label_column: str | None = None,
    out_dir: str | Path | None = None,
) -> CsvSplitReport:
    """Per-camera time-grouped stratified split.

    Like :func:`temporal_split_csv`, but sessions are formed *within each
    camera* — so identical timestamps from different cameras aren't merged, and
    a burst of close frames from one camera lands together in a single split.

    Args:
        csv_path: Path to the flat CVAT CSV export.
        camera_column: Camera id column.
        ts_column: Timestamp column (numeric epoch or ISO-8601 string).
        gap: Max spacing (in the ts unit) for two frames of one camera to stay
            in one session.
        ratios: Two or three fractions summing to 1.0.
        seed: RNG seed for reproducible assignment.
        stratify: Balance splits by each session's dominant class.
        label_column: Label column name; auto-detected when ``None``.
        out_dir: If given, write ``<split>.csv`` + ``split_manifest.csv`` here.

    Returns:
        A :class:`CsvSplitReport` (``assignment`` maps image_name → split).
    """
    _validate_ratios(ratios)
    records, header, rows = _read_image_records(
        csv_path,
        label_column=label_column,
        ts_column=ts_column,
        camera_column=camera_column,
    )
    if not records:
        raise ValueError(f"No images found in {csv_path}")
    _warn_missing_ts(records, ts_column)
    groups = _groups_camera_temporal(records, gap)
    return _finalize(
        records, groups, header, rows,
        ratios=ratios, seed=seed, stratify=stratify, out_dir=out_dir,
    )
