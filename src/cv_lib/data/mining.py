"""Hard-example mining — prioritise which unlabelled images to annotate.

Active-learning style ranking: score each unlabelled image by how *uncertain*
the current model is on it, then label the high-scoring ones first. The scoring
function is pure and unit-tested; :func:`rank_for_labeling` is the thin model
driver around it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def uncertainty_score(confs: list[float] | np.ndarray, by: str = "uncertainty") -> float:
    """Score an image's detection confidences — higher = more worth labelling.

    Args:
        confs: Per-detection confidence scores for one image.
        by: Strategy:
            ``"uncertainty"`` — mean closeness to 0.5 (boxes the model is least
                sure about); an image with no detections scores 1.0 (the model
                found nothing, likely a miss worth checking).
            ``"low_conf"`` — mean of ``1 - conf`` (favours weak detections).
            ``"num_detections"`` — raw detection count (busy / cluttered scenes).

    Returns:
        A non-negative score; callers rank images in descending order.
    """
    confs = np.asarray(list(confs), dtype=np.float64)
    if by == "num_detections":
        return float(confs.size)
    if confs.size == 0:
        return 1.0  # nothing detected → maximally worth a human look
    if by == "uncertainty":
        return float(np.mean(1.0 - np.abs(2.0 * confs - 1.0)))
    if by == "low_conf":
        return float(np.mean(1.0 - confs))
    raise ValueError(f"Unknown strategy {by!r}.")


def _load_model(model):
    if isinstance(model, (str, Path)):
        from ultralytics import YOLO

        return YOLO(str(model))
    return model


def rank_for_labeling(
    model,
    images_dir: str | Path,
    *,
    by: str = "uncertainty",
    conf: float = 0.05,
    imgsz: int = 640,
    device: str | None = None,
    extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
) -> list[tuple[Path, float]]:
    """Rank unlabelled images by labelling priority (most uncertain first).

    Args:
        model: Ultralytics ``YOLO`` instance or path to a ``.pt`` file.
        images_dir: Directory of images (searched recursively).
        by: Scoring strategy, see :func:`uncertainty_score`.
        conf: Low confidence floor so weak/uncertain boxes still count.
        imgsz: Inference image size.
        device: Optional device override.
        extensions: Image extensions to include.

    Returns:
        ``[(image_path, score), ...]`` sorted by descending score.
    """
    model = _load_model(model)
    images_dir = Path(images_dir)
    image_files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in extensions)

    predict_kwargs: dict = {"conf": conf, "imgsz": imgsz, "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device

    ranked: list[tuple[Path, float]] = []
    for img_path in image_files:
        results = model.predict(source=str(img_path), **predict_kwargs)
        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            cf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        else:
            cf = np.zeros(0, dtype=np.float32)
        ranked.append((img_path, uncertainty_score(cf, by)))

    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return ranked
