"""Tests for cv_lib.data.dedup (near-duplicate / leakage detection)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cv_lib.data.dedup import (
    check_split_leakage,
    dhash,
    find_duplicates,
    hamming,
)


def _gradient(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)


def test_hamming_and_dhash_identical(tmp_path: Path):
    img = _gradient(1)
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), img)
    assert dhash(p1) == dhash(p2)
    assert hamming(dhash(p1), dhash(p2)) == 0


def test_dhash_unreadable_returns_none(tmp_path: Path):
    bad = tmp_path / "nope.png"
    bad.write_bytes(b"not an image")
    assert dhash(bad) is None


def test_find_duplicates_clusters(tmp_path: Path):
    img = _gradient(7)
    cv2.imwrite(str(tmp_path / "orig.png"), img)
    cv2.imwrite(str(tmp_path / "copy.png"), img)  # exact duplicate
    cv2.imwrite(str(tmp_path / "other.png"), _gradient(99))  # distinct

    clusters = find_duplicates(tmp_path, hamming_threshold=5)
    assert len(clusters) == 1
    names = {p.name for p in clusters[0]}
    assert names == {"orig.png", "copy.png"}


def test_split_leakage(tmp_path: Path):
    img = _gradient(3)
    for split in ("train", "val"):
        d = tmp_path / "images" / split
        d.mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images" / "train" / "x.png"), img)
    cv2.imwrite(str(tmp_path / "images" / "val" / "y.png"), img)  # same image, other split
    cv2.imwrite(str(tmp_path / "images" / "train" / "z.png"), _gradient(123))

    pairs = check_split_leakage(tmp_path, hamming_threshold=5)
    assert len(pairs) == 1
    assert {pairs[0].split_a, pairs[0].split_b} == {"train", "val"}
