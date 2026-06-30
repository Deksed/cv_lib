"""Tests for predictions → CVAT CSV export and CVAT-CSV-sourced GT.

Covers cv_lib.data.convert.predictions_to_cvat_csv (reverse of cvat_csv_to_yolo),
the row/writer helpers, and cvat_csv_gt (per-image GT for visualization).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from cv_lib.data.convert import (
    CVAT_CSV_COLUMNS,
    _cvat_rows_for_image,
    _read_cvat_csv,
    _write_cvat_csv,
    cvat_csv_gt,
    predictions_to_cvat_csv,
    yolo_to_cvat_csv,
)


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf=None):
        self.xyxy = np.array(xyxy, dtype=float)
        self.cls = np.array(cls, dtype=float)
        self.conf = np.array(conf if conf is not None else [0.0] * len(cls), dtype=float)

    def __len__(self):
        return len(self.xyxy)


class _FakeModel:
    """One detection per image: class 0 at pixel box (10,20,110,140), 200x320 img."""

    names = {0: "car", 1: "person"}

    def predict(self, source, **kwargs):
        result = SimpleNamespace(
            boxes=_FakeBoxes([[10, 20, 110, 140]], [0.0], conf=[0.83]),
            masks=None,
            orig_shape=(200, 320),  # (h, w)
        )
        return [result]


class _FakeSegModel:
    """Seg model: same box plus a triangle mask via result.masks.xy."""

    names = {0: "car", 1: "person"}

    def predict(self, source, **kwargs):
        masks = SimpleNamespace(xy=[np.array([[10, 20], [110, 20], [60, 140]], dtype=float)])
        result = SimpleNamespace(
            boxes=_FakeBoxes([[10, 20, 110, 140]], [0.0], conf=[0.83]),
            masks=masks,
            orig_shape=(200, 320),
        )
        return [result]


def _make_images(d: Path, names: list[str]) -> None:
    d.mkdir(parents=True)
    for name in names:
        cv2.imwrite(str(d / name), np.full((200, 320, 3), 50, np.uint8))


# ---------------------------------------------------------------------------
# row / writer helpers
# ---------------------------------------------------------------------------

def test_cvat_rows_for_image_fields():
    columns = list(CVAT_CSV_COLUMNS)
    rows = _cvat_rows_for_image(
        "frame.jpg", 320, 200, [[10, 20, 110, 140]], ["car"], meta={}, columns=columns
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["image_name"] == "frame.jpg"
    assert row["image_width"] == "320"
    assert row["image_height"] == "200"
    assert row["instance_label"] == "car"
    assert row["instance_shape"] == "rectangle"
    assert row["instance_points"] == ""
    assert row["bbox_x_tl"] == "10.00"
    assert row["bbox_y_br"] == "140.00"


def test_write_and_read_roundtrip_with_template_extra(tmp_path: Path):
    # template carries bookkeeping + an extra "cvat_url" column
    columns = list(CVAT_CSV_COLUMNS) + ["cvat_url"]
    meta = {
        "image_id": "7",
        "task_id": "3",
        "task_name": "batch_3",
        "task_assignee": "alice",
        "image_path": "raw/frame.jpg",
        "cvat_url": "https://cvat/tasks/3/jobs/9",
    }
    rows = _cvat_rows_for_image(
        "frame.jpg", 320, 200, [[10, 20, 110, 140]], ["car"], meta=meta, columns=columns
    )
    out = tmp_path / "export.csv"
    _write_cvat_csv(rows, out, columns)

    header, read_rows = _read_cvat_csv(out)
    assert "cvat_url" in header
    assert len(read_rows) == 1
    r = read_rows[0]
    assert r["task_name"] == "batch_3"
    assert r["task_assignee"] == "alice"
    assert r["cvat_url"] == "https://cvat/tasks/3/jobs/9"
    # per-detection field is set by us, not the template:
    assert r["instance_label"] == "car"


# ---------------------------------------------------------------------------
# predictions_to_cvat_csv (fake model, no Ultralytics needed)
# ---------------------------------------------------------------------------

def test_predictions_to_cvat_csv_basic(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, ["a.jpg", "b.jpg"])
    out = tmp_path / "pred.csv"

    n = predictions_to_cvat_csv(_FakeModel(), images, out)
    assert n == 2  # one detection per image
    header, rows = _read_cvat_csv(out)
    assert header[: len(CVAT_CSV_COLUMNS)] == list(CVAT_CSV_COLUMNS)
    assert {r["image_name"] for r in rows} == {"a.jpg", "b.jpg"}
    assert all(r["instance_label"] == "car" for r in rows)  # class 0 via model.names
    assert all(r["image_width"] == "320" and r["image_height"] == "200" for r in rows)


def test_predictions_to_cvat_csv_joins_template(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, ["a.jpg"])

    template = tmp_path / "template.csv"
    columns = list(CVAT_CSV_COLUMNS) + ["cvat_url"]
    tmpl_row = {c: "" for c in columns}
    tmpl_row.update(
        {
            "image_name": "a.jpg",
            "task_name": "night_set",
            "task_assignee": "bob",
            "cvat_url": "https://cvat/jobs/42",
        }
    )
    _write_cvat_csv([tmpl_row], template, columns)

    out = tmp_path / "pred.csv"
    predictions_to_cvat_csv(_FakeModel(), images, out, template_csv=template)

    header, rows = _read_cvat_csv(out)
    assert "cvat_url" in header
    assert rows[0]["task_name"] == "night_set"
    assert rows[0]["task_assignee"] == "bob"
    assert rows[0]["cvat_url"] == "https://cvat/jobs/42"


# ---------------------------------------------------------------------------
# cvat_csv_gt
# ---------------------------------------------------------------------------

CSV_GT = """\
image_name,image_id,job_id,image_width,image_height,instance_label,instance_shape,instance_points,bbox_x_tl,bbox_y_tl,bbox_x_br,bbox_y_br,task_id,task_name,task_assignee,image_path,cvat_url
frame.jpg,1,9,320,200,car,rectangle,,10,20,110,140,3,batch,alice,raw/frame.jpg,https://cvat/9
frame.jpg,1,9,320,200,person,rectangle,,0,0,40,80,3,batch,alice,raw/frame.jpg,https://cvat/9
frame.jpg,1,9,320,200,tree,polygon,"5,5;60,5;30,40",0,0,0,0,3,batch,alice,raw/frame.jpg,https://cvat/9
"""


def test_cvat_csv_gt_parses_rectangles(tmp_path: Path):
    csv = tmp_path / "gt.csv"
    csv.write_text(CSV_GT, encoding="utf-8")
    records = cvat_csv_gt(csv, class_names=["car", "person"])

    assert set(records) == {"frame.jpg"}
    rec = records["frame.jpg"]
    assert rec["width"] == 320 and rec["height"] == 200
    assert rec["image_path"] == "raw/frame.jpg"
    # polygon row is dropped (rectangles only by default) → 2 boxes
    assert len(rec["boxes"]) == 2
    assert rec["class_ids"] == [0, 1]
    assert rec["labels"] == ["car", "person"]
    assert rec["boxes"][0] == [10.0, 20.0, 110.0, 140.0]
    # meta retains the extra cvat link column for the viz layer
    assert rec["meta"]["cvat_url"] == "https://cvat/9"
    # rectangle rows have no polygon
    assert rec["polygons"] == [None, None]


def test_cvat_csv_gt_includes_polygons_when_requested(tmp_path: Path):
    csv = tmp_path / "gt.csv"
    csv.write_text(CSV_GT, encoding="utf-8")
    records = cvat_csv_gt(
        csv, class_names=["car", "person", "tree"], shapes=("rectangle", "polygon")
    )
    rec = records["frame.jpg"]
    assert rec["labels"] == ["car", "person", "tree"]
    # polygon points parsed; its box derived from point bounds (bbox cols were 0)
    assert rec["polygons"][2] == [(5.0, 5.0), (60.0, 5.0), (30.0, 40.0)]
    assert rec["boxes"][2] == [5.0, 5.0, 60.0, 40.0]
    assert rec["polygons"][0] is None  # rectangle


# ---------------------------------------------------------------------------
# polygons (seg models) and confidence column
# ---------------------------------------------------------------------------

def test_predictions_to_cvat_csv_writes_polygons(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, ["a.jpg"])
    out = tmp_path / "seg.csv"

    predictions_to_cvat_csv(_FakeSegModel(), images, out, class_names=["car", "person"])
    _, rows = _read_cvat_csv(out)
    assert rows[0]["instance_shape"] == "polygon"
    assert rows[0]["instance_points"] == "10.00,20.00;110.00,20.00;60.00,140.00"
    # bbox still filled from the detection box
    assert rows[0]["bbox_x_tl"] == "10.00" and rows[0]["bbox_y_br"] == "140.00"


def test_predictions_to_cvat_csv_masks_false_keeps_rectangles(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, ["a.jpg"])
    out = tmp_path / "rect.csv"

    predictions_to_cvat_csv(
        _FakeSegModel(), images, out, class_names=["car", "person"], masks=False
    )
    _, rows = _read_cvat_csv(out)
    assert rows[0]["instance_shape"] == "rectangle"
    assert rows[0]["instance_points"] == ""


def test_predictions_to_cvat_csv_save_conf(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, ["a.jpg"])
    out = tmp_path / "pred.csv"

    predictions_to_cvat_csv(
        _FakeModel(), images, out, class_names=["car", "person"], save_conf=True
    )
    header, rows = _read_cvat_csv(out)
    assert "confidence" in header
    assert rows[0]["confidence"] == "0.8300"


# ---------------------------------------------------------------------------
# round-trip: predictions_to_cvat_csv → cvat_csv_gt
# ---------------------------------------------------------------------------

def test_export_then_read_back_roundtrip(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, ["a.jpg"])
    out = tmp_path / "pred.csv"

    predictions_to_cvat_csv(_FakeModel(), images, out, class_names=["car", "person"])
    records = cvat_csv_gt(out, class_names=["car", "person"])

    rec = records["a.jpg"]
    assert rec["width"] == 320 and rec["height"] == 200
    assert rec["labels"] == ["car"]
    assert rec["class_ids"] == [0]
    # the box the fake model emitted survives the write/parse round-trip
    assert rec["boxes"][0] == [10.0, 20.0, 110.0, 140.0]


# ---------------------------------------------------------------------------
# yolo_to_cvat_csv (GT export, counterpart of predictions_to_cvat_csv)
# ---------------------------------------------------------------------------

def test_yolo_to_cvat_csv_roundtrips_through_gt(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, ["a.jpg"])  # 200h x 320w
    labels = tmp_path / "labels"
    labels.mkdir()
    # one box: class 1, centre (0.5,0.5), w=0.25 h=0.5 → px (120,50,200,150)
    (labels / "a.txt").write_text("1 0.5 0.5 0.25 0.5\n")

    out = tmp_path / "gt.csv"
    n = yolo_to_cvat_csv(images, labels, out, class_names=["car", "person"])
    assert n == 1

    rec = cvat_csv_gt(out, class_names=["car", "person"])["a.jpg"]
    assert rec["labels"] == ["person"]
    assert rec["class_ids"] == [1]
    np.testing.assert_allclose(rec["boxes"][0], [120.0, 50.0, 200.0, 150.0], atol=1e-2)


def test_yolo_to_cvat_csv_skips_empty_labels(tmp_path: Path):
    images = tmp_path / "images"
    _make_images(images, ["a.jpg", "b.jpg"])
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (labels / "b.txt").write_text("")  # no boxes → no rows

    out = tmp_path / "gt.csv"
    n = yolo_to_cvat_csv(images, labels, out, class_names=["car"])
    assert n == 1
    _, rows = _read_cvat_csv(out)
    assert {r["image_name"] for r in rows} == {"a.jpg"}
