"""Annotation converters: CVAT XML and COCO JSON → YOLO .txt."""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# CVAT XML (YOLO 1.1 export) → YOLO txt
# ---------------------------------------------------------------------------

def cvat_xml_to_yolo(
    xml_path: str | Path,
    out_dir: str | Path,
    class_names: list[str] | None = None,
) -> dict[str, int]:
    """
    Convert CVAT XML annotation export to per-image YOLO .txt files.

    Supports CVAT's YOLO 1.1 XML format (<annotations> with <image> children
    and <box> elements carrying xtl/ytl/xbr/ybr attributes).

    Args:
        xml_path:     path to CVAT annotations.xml
        out_dir:      directory to write .txt files into
        class_names:  ordered class names; inferred from XML if None

    Returns:
        dict mapping class name → class index
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # build class map
    if class_names is None:
        seen: dict[str, int] = {}
        for image_el in root.findall("image"):
            for box in image_el.findall("box"):
                label = box.get("label", "")
                if label not in seen:
                    seen[label] = len(seen)
        class_map = seen
    else:
        class_map = {name: i for i, name in enumerate(class_names)}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_el in root.findall("image"):
        name = Path(image_el.get("name", "unknown")).stem
        img_w = float(image_el.get("width", 1))
        img_h = float(image_el.get("height", 1))

        lines: list[str] = []
        for box in image_el.findall("box"):
            label = box.get("label", "")
            if label not in class_map:
                continue
            cid = class_map[label]
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
            cx = ((xtl + xbr) / 2) / img_w
            cy = ((ytl + ybr) / 2) / img_h
            bw = (xbr - xtl) / img_w
            bh = (ybr - ytl) / img_h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out_dir / f"{name}.txt").write_text("\n".join(lines))

    return class_map


# ---------------------------------------------------------------------------
# COCO JSON → YOLO txt
# ---------------------------------------------------------------------------

def coco_json_to_yolo(
    json_path: str | Path,
    out_dir: str | Path,
    class_names: list[str] | None = None,
) -> dict[str, int]:
    """
    Convert a COCO-format JSON annotation file to per-image YOLO .txt files.

    Args:
        json_path:    path to COCO annotations JSON
        out_dir:      directory to write .txt files into
        class_names:  explicit ordered class name list; uses COCO categories if None

    Returns:
        dict mapping class name → class index (0-based)
    """
    import json

    data = json.loads(Path(json_path).read_text())

    # COCO category id → (name, 0-based index)
    categories = data.get("categories", [])
    if class_names is not None:
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        cat_id_to_idx: dict[int, int] = {}
        for cat in categories:
            name = cat["name"]
            if name in name_to_idx:
                cat_id_to_idx[cat["id"]] = name_to_idx[name]
    else:
        sorted_cats = sorted(categories, key=lambda c: c["id"])
        cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(sorted_cats)}
        name_to_idx = {cat["name"]: cat_id_to_idx[cat["id"]] for cat in sorted_cats}

    # image id → (filename stem, width, height)
    images_meta: dict[int, tuple[str, float, float]] = {}
    for img in data.get("images", []):
        stem = Path(img["file_name"]).stem
        images_meta[img["id"]] = (stem, float(img["width"]), float(img["height"]))

    # group annotations by image
    ann_by_image: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id, (stem, img_w, img_h) in images_meta.items():
        lines: list[str] = []
        for ann in ann_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_idx:
                continue
            cid = cat_id_to_idx[cat_id]
            x, y, bw, bh = ann["bbox"]  # COCO: x_min y_min width height
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (out_dir / f"{stem}.txt").write_text("\n".join(lines))

    return name_to_idx
