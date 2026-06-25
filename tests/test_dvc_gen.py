"""Tests for cv_lib.data.dvc_gen (DVC pipeline scaffolding, issue #3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cv_lib.data.dvc_gen import (
    STAGES,
    PipelineConfig,
    build_pipeline,
    default_params,
    generate_dvc_yaml,
    generate_params_yaml,
)


def test_build_pipeline_has_all_stages_in_canonical_order():
    pipeline = build_pipeline()
    assert list(pipeline["stages"]) == list(STAGES)
    for stage in pipeline["stages"].values():
        assert "cmd" in stage


def test_train_stage_tracks_params_and_config_threads_through():
    config = PipelineConfig(run_name="yolov8n_640")
    stages = build_pipeline(config)["stages"]
    assert stages["train"]["params"] == ["train"]
    # run_dir == runs_root/run_name should appear in train outs and report deps.
    assert "runs/train/yolov8n_640" in stages["train"]["outs"]
    assert any("yolov8n_640" in dep for dep in stages["report"]["deps"])


def test_build_pipeline_subset_keeps_canonical_order():
    pipeline = build_pipeline(stages=["report", "collect"])
    assert list(pipeline["stages"]) == ["collect", "report"]


def test_build_pipeline_rejects_unknown_stage():
    with pytest.raises(ValueError):
        build_pipeline(stages=["collect", "nope"])


def test_default_params_has_train_group():
    params = default_params()
    assert set(params["train"]) == {"model", "epochs", "imgsz", "batch", "seed"}


def test_generate_dvc_yaml_writes_valid_yaml(tmp_path: Path):
    out = generate_dvc_yaml(tmp_path / "dvc.yaml")
    assert out.exists()
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert list(loaded["stages"]) == list(STAGES)
    assert out.read_text(encoding="utf-8").startswith("#")  # header comment


def test_generate_refuses_existing_without_force(tmp_path: Path):
    target = tmp_path / "dvc.yaml"
    generate_dvc_yaml(target)
    with pytest.raises(FileExistsError):
        generate_dvc_yaml(target)
    # force overwrites cleanly
    assert generate_dvc_yaml(target, force=True) == target


def test_generate_params_yaml(tmp_path: Path):
    out = generate_params_yaml(tmp_path / "params.yaml")
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert loaded == default_params()
