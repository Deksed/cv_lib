"""Tests for Pascal VOC XML → YOLO import (cv_lib.data.convert.voc_to_yolo)."""

from __future__ import annotations

from pathlib import Path

from cv_lib.data.convert import voc_to_yolo

_VOC = """<annotation>
  <filename>frame.jpg</filename>
  <size><width>200</width><height>100</height><depth>3</depth></size>
  <object><name>car</name><bndbox><xmin>80</xmin><ymin>30</ymin><xmax>120</xmax><ymax>70</ymax></bndbox></object>
  <object><name>person</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object>
</annotation>
"""


def test_voc_to_yolo_basic(tmp_path: Path):
    voc = tmp_path / "voc"
    voc.mkdir()
    (voc / "frame.xml").write_text(_VOC)
    out = tmp_path / "labels"

    class_map = voc_to_yolo(voc, out, class_names=["car", "person"])
    assert class_map == {"car": 0, "person": 1}

    lines = (out / "frame.txt").read_text().splitlines()
    assert len(lines) == 2
    cid, cx, cy, w, h = lines[0].split()
    # car box: cx=(80+120)/2/200 = 0.5, w=40/200 = 0.2
    assert cid == "0"
    assert abs(float(cx) - 0.5) < 1e-6
    assert abs(float(w) - 0.2) < 1e-6


def test_voc_infers_class_names(tmp_path: Path):
    voc = tmp_path / "voc"
    voc.mkdir()
    (voc / "frame.xml").write_text(_VOC)
    out = tmp_path / "labels"
    class_map = voc_to_yolo(voc, out)
    assert class_map == {"car": 0, "person": 1}
