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
        "compare": ["compare", "img.jpg", "--model", "m.pt", "--names", "car"],
        "infer": ["infer", "--model", "m.pt", "--images", "imgs/"],
        "eval": ["eval", "--model", "m.pt", "--data", "d.yaml"],
        "bench": ["bench", "--model", "m.pt"],
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
