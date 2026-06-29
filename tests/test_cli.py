"""Tests for the unified cvlib CLI (cv_lib.cli)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cv_lib.cli import COMMANDS, build_parser, main


def test_parser_registers_all_commands():
    parser = build_parser()
    # Every registered command must be reachable as a subcommand and wired to its run().
    for name, module in COMMANDS.items():
        args = parser.parse_args(_minimal_args(name))
        assert args.command == name
        assert args._run is module.run


def _minimal_args(name: str) -> list[str]:
    """Smallest valid argv for each subcommand (just enough to parse)."""
    return {
        "inspect": ["inspect", "imgs/"],
        "convert": ["convert", "ann.json", "--out", "labels/"],
        "cvat-query": ["cvat-query", "export.csv", "--label", "car"],
        "compare": ["compare", "img.jpg", "--model", "m.pt", "--names", "car"],
        "infer": ["infer", "--model", "m.pt", "--images", "imgs/"],
        "eval": ["eval", "--model", "m.pt", "--data", "d.yaml"],
        "export": ["export", "m.pt"],
        "bench": ["bench", "--model", "m.pt"],
        "compare-runs": ["compare-runs", "runs/exp1"],
        "dvc-init": ["dvc-init"],
        "split": ["split", "imgs/", "--out", "ds/"],
        "distribution": ["distribution", "labels/"],
        "augment": ["augment", "img.jpg"],
        "remap": ["remap", "labels/", "--map", "1=0"],
        "qa": ["qa", "labels/"],
        "dedup": ["dedup", "imgs/"],
        "crops": ["crops", "imgs/"],
        "autolabel": ["autolabel", "imgs/", "--model", "m.pt", "--out", "labels/"],
        "mine": ["mine", "imgs/", "--model", "m.pt"],
        "threshold": ["threshold", "--model", "m.pt", "--data", "d.yaml"],
    }[name]


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "cvlib" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    code = main([])
    assert code == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_convert_coco_end_to_end(tmp_path: Path):
    coco = {
        "images": [{"id": 1, "file_name": "frame.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10]},
        ],
    }
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(coco))
    out_dir = tmp_path / "labels"

    code = main(["convert", str(json_path), "--out", str(out_dir)])
    assert code == 0

    label = out_dir / "frame.txt"
    assert label.exists()
    lines = label.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("0 ")  # car → index 0
    assert lines[1].startswith("1 ")  # person → index 1


def test_convert_format_inference_error(tmp_path: Path):
    bad = tmp_path / "ann.txt"
    bad.write_text("")
    with pytest.raises(SystemExit):
        main(["convert", str(bad), "--out", str(tmp_path / "out")])


def test_compare_runs_end_to_end(tmp_path: Path, capsys):
    run_dir = tmp_path / "exp1"
    run_dir.mkdir()
    (run_dir / "train_config.json").write_text(
        json.dumps({"model_path": "yolov8n.pt", "epochs": 50, "imgsz": 640})
    )
    (run_dir / "results.csv").write_text(
        "epoch, metrics/mAP50(B), metrics/mAP50-95(B)\n"
        "1, 0.5, 0.3\n"
        "2, 0.8, 0.6\n"
    )

    code = main(["compare-runs", str(run_dir)])
    assert code == 0

    out = capsys.readouterr().out
    assert "yolov8n.pt" in out
    assert "0.8000" in out  # best mAP50 row selected


def test_compare_runs_missing_dir_exits(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["compare-runs", str(tmp_path / "does_not_exist")])


_CVAT_CSV = (
    "image_name,image_id,job_id,image_width,image_height,instance_label,"
    "instance_shape,instance_points,bbox_x_tl,bbox_y_tl,bbox_x_br,bbox_y_br,"
    "task_id,task_name,task_assignee,image_path\n"
    "frame_001.jpg,0,5,640,480,car,rectangle,,100,50,300,200,1,batch_3,anna,images/frame_001.jpg\n"
    "frame_002.jpg,1,5,320,240,person,rectangle,,10,20,60,120,1,batch_3,bob,images/frame_002.jpg\n"
)


def test_convert_cvat_csv_end_to_end(tmp_path: Path):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(_CVAT_CSV)
    out_dir = tmp_path / "labels"

    code = main(["convert", str(csv_path), "--out", str(out_dir)])
    assert code == 0
    assert (out_dir / "frame_001.txt").exists()
    assert (out_dir / "frame_002.txt").read_text().startswith("1 ")  # person → idx 1


def test_cvat_query_count(tmp_path: Path, capsys):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(_CVAT_CSV)

    code = main(["cvat-query", str(csv_path), "--assignee", "anna", "--count"])
    assert code == 0
    assert capsys.readouterr().out.strip() == "1"


def test_dvc_init_end_to_end(tmp_path: Path):
    import yaml

    dvc = tmp_path / "dvc.yaml"
    params = tmp_path / "params.yaml"

    code = main(["dvc-init", "--out", str(dvc), "--params-out", str(params)])
    assert code == 0
    assert dvc.exists() and params.exists()

    stages = yaml.safe_load(dvc.read_text(encoding="utf-8"))["stages"]
    assert "train" in stages and "report" in stages


def test_dvc_init_refuses_existing(tmp_path: Path):
    dvc = tmp_path / "dvc.yaml"
    assert main(["dvc-init", "--out", str(dvc), "--no-params"]) == 0
    with pytest.raises(SystemExit):
        main(["dvc-init", "--out", str(dvc), "--no-params"])


def test_distribution_end_to_end(tmp_path: Path, capsys):
    # Dataset root: labels/<split> subdirs + a data.yaml for class names.
    for split, seq in (("train", [0, 0, 1]), ("val", [0, 1, 1])):
        d = tmp_path / "labels" / split
        d.mkdir(parents=True)
        for i, cid in enumerate(seq):
            (d / f"f_{i}.txt").write_text(f"{cid} 0.5 0.5 0.2 0.2\n")
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: [car, person]\n")
    out = tmp_path / "dist.png"

    code = main(["distribution", str(tmp_path), "--out", str(out)])
    assert code == 0
    assert out.exists()

    printed = capsys.readouterr().out
    assert "car" in printed and "person" in printed
    assert "TOTAL" in printed


def test_augment_end_to_end(tmp_path: Path, capsys):
    import cv2
    import numpy as np

    img = tmp_path / "frame.jpg"
    cv2.imwrite(str(img), np.zeros((80, 80, 3), dtype=np.uint8))
    (tmp_path / "frame.txt").write_text("0 0.5 0.5 0.4 0.3\n")  # sibling label, auto-detected
    out = tmp_path / "aug.png"

    code = main(["augment", str(img), "--names", "car", "-n", "3", "--out", str(out)])
    assert code == 0
    assert out.exists()
    assert "Saved augmentation preview" in capsys.readouterr().out


def test_augment_missing_image_exits(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["augment", str(tmp_path / "nope.jpg")])


def test_split_end_to_end(tmp_path: Path):
    import yaml

    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(10):
        (images / f"f_{i}.jpg").write_bytes(b"x")
        (labels / f"f_{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.2 0.2\n")
    out = tmp_path / "ds"

    code = main(["split", str(images), "--labels", str(labels), "--out", str(out),
                 "--ratios", "0.8", "0.2", "--names", "car", "person"])
    assert code == 0

    doc = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert doc["names"] == ["car", "person"]
    assert (out / "images" / "train").is_dir()
    assert not (out / "images" / "test").exists()  # two-way split
