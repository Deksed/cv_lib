"""Tests for cv_lib.data.qa (annotation anomaly audit)."""

from __future__ import annotations

from pathlib import Path

from cv_lib.data.qa import audit_labels


def test_flags_tiny_and_huge(tmp_path: Path):
    (tmp_path / "a.txt").write_text(
        "0 0.5 0.5 0.001 0.001\n"  # tiny: area 1e-6
        "1 0.5 0.5 0.99 0.99\n"  # huge: area ~0.98
        "2 0.5 0.5 0.2 0.2\n"  # normal
    )
    report = audit_labels(tmp_path)
    kinds = report.by_kind()
    assert kinds.get("tiny") == 1
    assert kinds.get("huge") == 1
    assert report.boxes_checked == 3


def test_flags_extreme_aspect(tmp_path: Path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.5 0.02\n")  # aspect 25
    report = audit_labels(tmp_path, max_aspect=10.0)
    assert report.by_kind().get("aspect") == 1


def test_flags_duplicate_boxes(tmp_path: Path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
    report = audit_labels(tmp_path)
    assert report.by_kind().get("duplicate") == 1


def test_flags_count_outlier(tmp_path: Path):
    for i in range(6):
        (tmp_path / f"f{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    # one file with a huge object count
    (tmp_path / "outlier.txt").write_text("\n".join("0 0.5 0.5 0.05 0.05" for _ in range(50)))
    report = audit_labels(tmp_path, count_z=2.0)
    outliers = [f for f in report.findings if f.kind == "count_outlier"]
    assert any(f.file == "outlier.txt" for f in outliers)


def test_clean_dataset_no_findings(tmp_path: Path):
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.15 0.15\n")
    (tmp_path / "b.txt").write_text("0 0.4 0.4 0.2 0.25\n")
    report = audit_labels(tmp_path)
    assert report.findings == []
