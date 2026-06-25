"""Tests for cv_lib.data.convert — CVAT XML and COCO JSON → YOLO txt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cv_lib.data.convert import (
    coco_json_to_yolo,
    cvat_csv_to_yolo,
    cvat_xml_to_yolo,
    query_cvat_csv,
)

# ---------------------------------------------------------------------------
# CVAT XML
# ---------------------------------------------------------------------------

CVAT_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="images/frame_001.jpg" width="640" height="480">
    <box label="car" xtl="100" ytl="50" xbr="300" ybr="200" occluded="0"/>
    <box label="person" xtl="10" ytl="20" xbr="60" ybr="120" occluded="0"/>
  </image>
  <image id="1" name="frame_002.jpg" width="320" height="240">
    <box label="car" xtl="0" ytl="0" xbr="100" ybr="100" occluded="0"/>
  </image>
</annotations>
"""


def test_cvat_xml_creates_files(tmp_path: Path):
    xml = tmp_path / "ann.xml"
    xml.write_text(CVAT_XML)
    out = tmp_path / "labels"
    class_map = cvat_xml_to_yolo(xml, out, class_names=["car", "person"])
    assert (out / "frame_001.txt").exists()
    assert (out / "frame_002.txt").exists()
    assert class_map == {"car": 0, "person": 1}


def test_cvat_xml_yolo_coords(tmp_path: Path):
    xml = tmp_path / "ann.xml"
    xml.write_text(CVAT_XML)
    out = tmp_path / "labels"
    cvat_xml_to_yolo(xml, out, class_names=["car", "person"])

    lines = (out / "frame_001.txt").read_text().strip().splitlines()
    assert len(lines) == 2

    # car: xtl=100 ytl=50 xbr=300 ybr=200, image 640×480
    # cx=(100+300)/2/640=0.3125, cy=(50+200)/2/480=0.2604, w=200/640=0.3125, h=150/480=0.3125
    parts = lines[0].split()
    assert parts[0] == "0"
    assert abs(float(parts[1]) - 0.3125) < 1e-4
    assert abs(float(parts[2]) - (125 / 480)) < 1e-4


def test_cvat_xml_infer_classes(tmp_path: Path):
    xml = tmp_path / "ann.xml"
    xml.write_text(CVAT_XML)
    out = tmp_path / "labels"
    class_map = cvat_xml_to_yolo(xml, out)
    assert "car" in class_map
    assert "person" in class_map


# ---------------------------------------------------------------------------
# COCO JSON
# ---------------------------------------------------------------------------

COCO_DATA = {
    "categories": [
        {"id": 1, "name": "cat"},
        {"id": 2, "name": "dog"},
    ],
    "images": [
        {"id": 10, "file_name": "img_001.jpg", "width": 400, "height": 300},
    ],
    "annotations": [
        {"id": 1, "image_id": 10, "category_id": 1, "bbox": [40, 30, 80, 60]},
        {"id": 2, "image_id": 10, "category_id": 2, "bbox": [200, 100, 50, 50]},
    ],
}


def test_coco_json_creates_file(tmp_path: Path):
    jf = tmp_path / "ann.json"
    jf.write_text(json.dumps(COCO_DATA))
    out = tmp_path / "labels"
    coco_json_to_yolo(jf, out)
    assert (out / "img_001.txt").exists()


def test_coco_json_yolo_coords(tmp_path: Path):
    jf = tmp_path / "ann.json"
    jf.write_text(json.dumps(COCO_DATA))
    out = tmp_path / "labels"
    name_map = coco_json_to_yolo(jf, out)

    lines = (out / "img_001.txt").read_text().strip().splitlines()
    assert len(lines) == 2

    # cat bbox [40,30,80,60] in 400×300
    # cx=(40+40)/400=0.2, cy=(30+30)/300=0.2, w=80/400=0.2, h=60/300=0.2
    cat_idx = name_map["cat"]
    cat_line = next(l for l in lines if l.startswith(str(cat_idx)))
    parts = cat_line.split()
    assert abs(float(parts[1]) - 0.2) < 1e-4
    assert abs(float(parts[2]) - 0.2) < 1e-4
    assert abs(float(parts[3]) - 0.2) < 1e-4


def test_coco_json_custom_class_names(tmp_path: Path):
    jf = tmp_path / "ann.json"
    jf.write_text(json.dumps(COCO_DATA))
    out = tmp_path / "labels"
    name_map = coco_json_to_yolo(jf, out, class_names=["dog", "cat"])
    assert name_map["dog"] == 0
    assert name_map["cat"] == 1


# ---------------------------------------------------------------------------
# CVAT CSV
# ---------------------------------------------------------------------------

CVAT_CSV = """\
image_name,image_id,job_id,image_width,image_height,instance_label,instance_shape,instance_points,bbox_x_tl,bbox_y_tl,bbox_x_br,bbox_y_br,task_id,task_name,task_assignee,image_path
frame_001.jpg,0,5,640,480,car,rectangle,,100,50,300,200,1,batch_3,anna,images/frame_001.jpg
frame_001.jpg,0,5,640,480,person,rectangle,,10,20,60,120,1,batch_3,anna,images/frame_001.jpg
frame_002.jpg,1,5,320,240,car,rectangle,,0,0,100,100,1,batch_3,bob,images/frame_002.jpg
frame_002.jpg,1,5,320,240,tree,polygon,"5,5;10,10;5,10",0,0,50,50,1,batch_3,bob,images/frame_002.jpg
"""


def test_cvat_csv_creates_files(tmp_path: Path):
    csv_file = tmp_path / "export.csv"
    csv_file.write_text(CVAT_CSV)
    out = tmp_path / "labels"
    class_map = cvat_csv_to_yolo(csv_file, out, class_names=["car", "person"])
    assert (out / "frame_001.txt").exists()
    assert (out / "frame_002.txt").exists()
    assert class_map == {"car": 0, "person": 1}


def test_cvat_csv_yolo_coords(tmp_path: Path):
    csv_file = tmp_path / "export.csv"
    csv_file.write_text(CVAT_CSV)
    out = tmp_path / "labels"
    cvat_csv_to_yolo(csv_file, out, class_names=["car", "person"])

    lines = (out / "frame_001.txt").read_text().strip().splitlines()
    assert len(lines) == 2
    # car: xtl=100 ytl=50 xbr=300 ybr=200 in 640×480 (same maths as the XML case)
    parts = lines[0].split()
    assert parts[0] == "0"
    assert abs(float(parts[1]) - 0.3125) < 1e-4
    assert abs(float(parts[2]) - (125 / 480)) < 1e-4


def test_cvat_csv_infers_classes_and_skips_polygons(tmp_path: Path):
    csv_file = tmp_path / "export.csv"
    csv_file.write_text(CVAT_CSV)
    out = tmp_path / "labels"
    class_map = cvat_csv_to_yolo(csv_file, out)  # default shapes=("rectangle",)
    # 'tree' is polygon-only → excluded by the rectangle shape filter
    assert "tree" not in class_map
    assert class_map == {"car": 0, "person": 1}
    # frame_002 keeps only the rectangle 'car', not the polygon 'tree'
    assert len((out / "frame_002.txt").read_text().strip().splitlines()) == 1


def test_cvat_csv_all_shapes(tmp_path: Path):
    csv_file = tmp_path / "export.csv"
    csv_file.write_text(CVAT_CSV)
    out = tmp_path / "labels"
    class_map = cvat_csv_to_yolo(csv_file, out, shapes=None)
    assert "tree" in class_map


def test_query_cvat_csv_filters(tmp_path: Path):
    csv_file = tmp_path / "export.csv"
    csv_file.write_text(CVAT_CSV)

    cars = query_cvat_csv(csv_file, instance_label="car")
    assert len(cars) == 2

    annas_cars = query_cvat_csv(csv_file, instance_label="car", task_assignee="anna")
    assert len(annas_cars) == 1
    assert annas_cars[0]["image_name"] == "frame_001.jpg"


def test_query_cvat_csv_unknown_column(tmp_path: Path):
    csv_file = tmp_path / "export.csv"
    csv_file.write_text(CVAT_CSV)
    with pytest.raises(ValueError):
        query_cvat_csv(csv_file, nope="x")
