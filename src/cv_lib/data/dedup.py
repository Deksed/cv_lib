"""Near-duplicate image detection and train/val/test data-leakage checks.

Uses a difference hash (dHash): downscale to grayscale, compare adjacent pixel
brightness, pack the booleans into a 64-bit integer. Similar images produce
hashes within a small Hamming distance — no extra dependency beyond OpenCV.

Duplicate images leaking across splits is a classic cause of unrealistically
good validation metrics; :func:`check_split_leakage` finds those cross-split
near-duplicates explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def dhash(image_path: str | Path, hash_size: int = 8) -> int | None:
    """Compute the difference hash of an image as an integer (``None`` if unreadable)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def hamming(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return bin(a ^ b).count("1")


def _collect_images(images_dir: str | Path) -> list[Path]:
    return sorted(
        p for p in Path(images_dir).rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _hash_images(paths: list[Path], hash_size: int) -> list[tuple[Path, int]]:
    out: list[tuple[Path, int]] = []
    for p in paths:
        h = dhash(p, hash_size)
        if h is not None:
            out.append((p, h))
    return out


def find_duplicates(
    images_dir: str | Path,
    *,
    hamming_threshold: int = 5,
    hash_size: int = 8,
) -> list[list[Path]]:
    """Group near-duplicate images within a directory.

    Args:
        images_dir: Directory of images (searched recursively).
        hamming_threshold: Max Hamming distance between dHashes to call two
            images duplicates (0 = identical hash; ~5 tolerates light edits).
        hash_size: dHash grid size (8 → 64-bit hash).

    Returns:
        Clusters (lists) of paths that are mutual near-duplicates; singletons
        are omitted. Uses single-linkage grouping (union-find).
    """
    hashed = _hash_images(_collect_images(images_dir), hash_size)
    n = len(hashed)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if hamming(hashed[i][1], hashed[j][1]) <= hamming_threshold:
                union(i, j)

    clusters: dict[int, list[Path]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(hashed[i][0])
    return [sorted(group) for group in clusters.values() if len(group) > 1]


@dataclass
class LeakagePair:
    """A near-duplicate image found in two different splits."""

    path_a: Path
    split_a: str
    path_b: Path
    split_b: str
    distance: int


def check_split_leakage(
    dataset_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    hamming_threshold: int = 5,
    hash_size: int = 8,
) -> list[LeakagePair]:
    """Find near-duplicate images that span two different splits.

    Expects the YOLO layout ``<dataset_dir>/images/<split>/``. Splits that are
    missing on disk are skipped.

    Returns:
        A list of :class:`LeakagePair`, one per cross-split near-duplicate.
    """
    dataset_dir = Path(dataset_dir)
    split_hashes: dict[str, list[tuple[Path, int]]] = {}
    for split in splits:
        split_dir = dataset_dir / "images" / split
        if split_dir.is_dir():
            split_hashes[split] = _hash_images(_collect_images(split_dir), hash_size)

    pairs: list[LeakagePair] = []
    names = list(split_hashes)
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            sa, sb = names[a_i], names[b_i]
            for pa, ha in split_hashes[sa]:
                for pb, hb in split_hashes[sb]:
                    d = hamming(ha, hb)
                    if d <= hamming_threshold:
                        pairs.append(LeakagePair(pa, sa, pb, sb, d))
    return pairs
