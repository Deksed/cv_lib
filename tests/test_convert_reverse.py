"""Tests for YOLO → COCO / VOC export (cv_lib.data.convert reverse direction)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from cv_lib.data.convert import coco_json_to_yolo, yolo_to_coco, yolo_to_voc


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cv2.imwrite(str(images / "frame.jpg"), np.full((100, 200, 3), 64, np.uint8))
    (labels / "frame.txt").write_text("0 0.5 0.5 0.2 0.4\n1 0.25 0.25 0.1 0.1\n")
    return images, labels


def test_yolo_to_coco_structure(tmp_path: Path):
    images, labels = _setup(tmp_path)
    out = tmp_path / "ann.json"

    coco = yolo_to_coco(images, labels, out, ["car", "person"])
    assert out.exists()
    assert len(coco["images"]) == 1
    assert coco["images"][0]["width"] == 200 and coco["images"][0]["height"] == 100
    assert len(coco["annotations"]) == 2
    # class 0 box: cx=0.5*200=100, w=0.2*200=40 → x = 100-20 = 80
    car = next(a for a in coco["annotations"] if a["category_id"] == 1)
    assert car["bbox"][0] == 80.0
    assert car["bbox"][2] == 40.0  # width in px


def test_yolo_coco_roundtrip(tmp_path: Path):
    images, labels = _setup(tmp_path)
    out = tmp_path / "ann.json"
    yolo_to_coco(images, labels, out, ["car", "person"])

    back = tmp_path / "back"
    coco_json_to_yolo(out, back, class_names=["car", "person"])
    orig = (labels / "frame.txt").read_text().split()
    restored = (back / "frame.txt").read_text().split()
    # class ids and normalised coords survive the round-trip
    assert restored[0] == orig[0] == "0"
    assert abs(float(restored[1]) - float(orig[1])) < 1e-4


def test_yolo_to_voc(tmp_path: Path):
    images, labels = _setup(tmp_path)
    out = tmp_path / "voc"
    n = yolo_to_voc(images, labels, out, ["car", "person"])
    assert n == 1
    xml = ET.parse(out / "frame.xml").getroot()
    assert xml.findtext("size/width") == "200"
    objs = xml.findall("object")
    assert {o.findtext("name") for o in objs} == {"car", "person"}


def test_coco_written_is_valid_json(tmp_path: Path):
    images, labels = _setup(tmp_path)
    out = tmp_path / "ann.json"
    yolo_to_coco(images, labels, out, ["car", "person"])
    json.loads(out.read_text())  # must not raise
