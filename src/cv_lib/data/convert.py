"""Annotation converters: CVAT XML / COCO JSON / CVAT CSV → YOLO .txt, and the
reverse YOLO → COCO JSON / Pascal VOC XML for interop and submissions."""

from __future__ import annotations

from pathlib import Path

_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

# CVAT CSV export columns (one row per annotated instance).
# See issue #2: a flat export with bbox corners + task/job metadata.
CVAT_CSV_COLUMNS = (
    "image_name",
    "image_id",
    "job_id",
    "image_width",
    "image_height",
    "instance_label",
    "instance_shape",
    "instance_points",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
    "task_id",
    "task_name",
    "task_assignee",
    "image_path",
)


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


# ---------------------------------------------------------------------------
# Pascal VOC XML → YOLO txt
# ---------------------------------------------------------------------------

def voc_to_yolo(
    voc_dir: str | Path,
    out_dir: str | Path,
    class_names: list[str] | None = None,
) -> dict[str, int]:
    """Convert a directory of Pascal VOC XML annotations to YOLO ``.txt`` files.

    Each VOC file carries ``<size>`` (width/height) and ``<object>`` entries with
    a ``<name>`` and ``<bndbox>`` (xmin/ymin/xmax/ymax in pixels). One ``.txt`` is
    written per XML.

    Args:
        voc_dir:      directory of ``*.xml`` VOC annotation files
        out_dir:      directory to write ``.txt`` files into
        class_names:  ordered class names; inferred (first-seen) from the XML if None

    Returns:
        dict mapping class name → class index
    """
    import xml.etree.ElementTree as ET

    voc_dir = Path(voc_dir)
    xml_files = sorted(voc_dir.glob("*.xml"))

    if class_names is None:
        class_map: dict[str, int] = {}
        for xml_path in xml_files:
            for obj in ET.parse(xml_path).getroot().findall("object"):
                name = (obj.findtext("name") or "").strip()
                if name and name not in class_map:
                    class_map[name] = len(class_map)
    else:
        class_map = {name: i for i, name in enumerate(class_names)}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        img_w = float(size.findtext("width", "1")) if size is not None else 1.0
        img_h = float(size.findtext("height", "1")) if size is not None else 1.0
        if img_w <= 0 or img_h <= 0:
            continue

        lines: list[str] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in class_map:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            bw = (xmax - xmin) / img_w
            bh = (ymax - ymin) / img_h
            lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        stem = root.findtext("filename") or xml_path.name
        (out_dir / f"{Path(stem).stem}.txt").write_text("\n".join(lines))

    return class_map


# ---------------------------------------------------------------------------
# CVAT CSV (flat per-instance export) → YOLO txt
# ---------------------------------------------------------------------------

def _read_cvat_csv(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CVAT CSV export into (header, rows), stripping whitespace."""
    import csv

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    return header, rows


def cvat_csv_to_yolo(
    csv_path: str | Path,
    out_dir: str | Path,
    class_names: list[str] | None = None,
    shapes: tuple[str, ...] | None = ("rectangle",),
) -> dict[str, int]:
    """
    Convert a flat CVAT CSV export to per-image YOLO .txt files.

    The CSV holds one row per annotated instance with bbox corners
    (``bbox_x_tl``/``bbox_y_tl``/``bbox_x_br``/``bbox_y_br``) and per-image
    dimensions (``image_width``/``image_height``). Rows are grouped by image;
    one .txt is written per image (empty if it has no usable boxes).

    Args:
        csv_path:     path to the CVAT CSV export
        out_dir:      directory to write .txt files into
        class_names:  ordered class names; inferred from ``instance_label`` if None
        shapes:       keep only these ``instance_shape`` values (case-insensitive);
                      None keeps every row that has a valid bbox

    Returns:
        dict mapping class name → class index
    """
    _, rows = _read_cvat_csv(csv_path)

    shape_set = {s.lower() for s in shapes} if shapes is not None else None

    def _shape_ok(row: dict[str, str]) -> bool:
        return shape_set is None or row.get("instance_shape", "").lower() in shape_set

    # Build class map: explicit names, or inferred (first-seen order) from the
    # rows that pass the shape filter — so polygon-only labels don't leak in.
    if class_names is None:
        class_map: dict[str, int] = {}
        for row in rows:
            label = row.get("instance_label", "")
            if label and _shape_ok(row) and label not in class_map:
                class_map[label] = len(class_map)
    else:
        class_map = {name: i for i, name in enumerate(class_names)}

    # Group rows by image; remember each image's stem and dimensions.
    images: dict[str, dict] = {}
    for row in rows:
        name = row.get("image_name") or row.get("image_path")
        if not name:
            continue
        stem = Path(name).stem
        entry = images.setdefault(stem, {"lines": []})

        if not _shape_ok(row):
            continue

        label = row.get("instance_label", "")
        if label not in class_map:
            continue

        try:
            img_w = float(row["image_width"])
            img_h = float(row["image_height"])
            xtl = float(row["bbox_x_tl"])
            ytl = float(row["bbox_y_tl"])
            xbr = float(row["bbox_x_br"])
            ybr = float(row["bbox_y_br"])
        except (KeyError, ValueError):
            continue
        if img_w <= 0 or img_h <= 0:
            continue

        cid = class_map[label]
        cx = ((xtl + xbr) / 2) / img_w
        cy = ((ytl + ybr) / 2) / img_h
        bw = (xbr - xtl) / img_w
        bh = (ybr - ytl) / img_h
        entry["lines"].append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, entry in images.items():
        (out_dir / f"{stem}.txt").write_text("\n".join(entry["lines"]))

    return class_map


def cvat_csv_gt(
    csv_path: str | Path,
    *,
    class_names: list[str] | None = None,
    shapes: tuple[str, ...] | None = ("rectangle",),
) -> dict[str, dict]:
    """Parse a CVAT CSV export into per-image ground truth for visualization.

    Unlike :func:`cvat_csv_to_yolo` (which writes YOLO ``.txt`` keyed by stem),
    this keeps the CSV as the source of truth — including ``image_path`` and any
    extra columns (e.g. a CVAT link) — so callers can draw/compare straight from
    the export instead of a YOLO dataset.

    Args:
        csv_path:    path to the CVAT CSV export
        class_names: ordered class names; inferred (first-seen) from
            ``instance_label`` if None
        shapes:      keep only these ``instance_shape`` values (case-insensitive);
            None keeps every row with a valid bbox

    Returns:
        dict mapping ``image_name`` → record with keys ``image_path`` (str),
        ``width``/``height`` (int), ``boxes`` (list of ``[x1, y1, x2, y2]`` px),
        ``class_ids`` (list[int]), ``labels`` (list[str]) and ``meta`` (the first
        row for that image, carrying bookkeeping/extra columns).
    """
    _, rows = _read_cvat_csv(csv_path)

    shape_set = {s.lower() for s in shapes} if shapes is not None else None

    def _shape_ok(row: dict[str, str]) -> bool:
        return shape_set is None or row.get("instance_shape", "").lower() in shape_set

    if class_names is None:
        class_map: dict[str, int] = {}
        for row in rows:
            label = row.get("instance_label", "")
            if label and _shape_ok(row) and label not in class_map:
                class_map[label] = len(class_map)
    else:
        class_map = {name: i for i, name in enumerate(class_names)}

    records: dict[str, dict] = {}
    for row in rows:
        name = row.get("image_name") or row.get("image_path")
        if not name:
            continue
        name = Path(name).name
        rec = records.get(name)
        if rec is None:
            try:
                width = int(float(row.get("image_width", "0") or 0))
                height = int(float(row.get("image_height", "0") or 0))
            except ValueError:
                width = height = 0
            rec = {
                "image_path": row.get("image_path") or name,
                "width": width,
                "height": height,
                "boxes": [],
                "class_ids": [],
                "labels": [],
                "meta": row,
            }
            records[name] = rec

        if not _shape_ok(row):
            continue
        label = row.get("instance_label", "")
        if label not in class_map:
            continue
        try:
            xtl = float(row["bbox_x_tl"])
            ytl = float(row["bbox_y_tl"])
            xbr = float(row["bbox_x_br"])
            ybr = float(row["bbox_y_br"])
        except (KeyError, ValueError):
            continue
        rec["boxes"].append([xtl, ytl, xbr, ybr])
        rec["class_ids"].append(class_map[label])
        rec["labels"].append(label)

    return records


def query_cvat_csv(
    csv_path: str | Path,
    **filters: str,
) -> list[dict[str, str]]:
    """
    Filter rows of a CVAT CSV export by exact column matches.

    Example::

        query_cvat_csv("export.csv", task_name="batch_3", instance_label="car")

    Args:
        csv_path:  path to the CVAT CSV export
        filters:   column=value pairs; a row matches when every column equals
                   the given value (compared as strings)

    Returns:
        list of matching rows (each a dict of stripped column → value)

    Raises:
        ValueError: if a filter names a column not present in the CSV header
    """
    header, rows = _read_cvat_csv(csv_path)

    unknown = [k for k in filters if k not in header]
    if unknown:
        raise ValueError(
            f"Unknown column(s) {unknown}; available columns: {header}"
        )

    wanted = {k: str(v) for k, v in filters.items()}
    return [row for row in rows if all(row.get(k) == v for k, v in wanted.items())]


# ---------------------------------------------------------------------------
# YOLO txt → COCO JSON / Pascal VOC XML (reverse direction)
# ---------------------------------------------------------------------------

def _image_size(img_path: Path) -> tuple[int, int]:
    """Return (width, height) of an image, reading the file with OpenCV."""
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image {img_path}")
    h, w = img.shape[:2]
    return w, h


def _iter_yolo_pairs(images_dir: Path, labels_dir: Path | None):
    """Yield (image_path, label_path) pairs (label may not exist)."""
    from cv_lib.data import iter_image_label_pairs

    return iter_image_label_pairs(images_dir, labels_dir, _IMAGE_EXTENSIONS)


def _read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append((cid, cx, cy, w, h))
    return boxes


def yolo_to_coco(
    images_dir: str | Path,
    labels_dir: str | Path | None,
    out_json: str | Path,
    class_names: list[str],
) -> dict:
    """Convert a YOLO dataset to a single COCO-format JSON file.

    Args:
        images_dir: Directory of images.
        labels_dir: Directory of YOLO ``.txt`` labels (inferred from
            ``images_dir`` if ``None``).
        out_json: Destination ``.json`` path.
        class_names: Ordered class names (index = YOLO class id).

    Returns:
        The COCO dict that was written.
    """
    import json

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir is not None else None

    coco: dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }
    ann_id = 1
    for img_id, (img_path, label_path) in enumerate(_iter_yolo_pairs(images_dir, labels_dir), 1):
        iw, ih = _image_size(img_path)
        coco["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": iw, "height": ih}
        )
        for cid, cx, cy, w, h in _read_yolo_boxes(label_path):
            bw, bh = w * iw, h * ih
            x = (cx * iw) - bw / 2
            y = (cy * ih) - bh / 2
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid + 1,  # COCO ids are 1-based
                    "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, indent=2))
    return coco


def yolo_to_voc(
    images_dir: str | Path,
    labels_dir: str | Path | None,
    out_dir: str | Path,
    class_names: list[str],
) -> int:
    """Convert a YOLO dataset to per-image Pascal VOC XML files.

    Args:
        images_dir: Directory of images.
        labels_dir: Directory of YOLO ``.txt`` labels (inferred if ``None``).
        out_dir: Directory to write ``<stem>.xml`` files into.
        class_names: Ordered class names (index = YOLO class id).

    Returns:
        Number of XML files written.
    """
    import xml.etree.ElementTree as ET

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir is not None else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for img_path, label_path in _iter_yolo_pairs(images_dir, labels_dir):
        iw, ih = _image_size(img_path)
        ann = ET.Element("annotation")
        ET.SubElement(ann, "filename").text = img_path.name
        size = ET.SubElement(ann, "size")
        ET.SubElement(size, "width").text = str(iw)
        ET.SubElement(size, "height").text = str(ih)
        ET.SubElement(size, "depth").text = "3"
        for cid, cx, cy, w, h in _read_yolo_boxes(label_path):
            bw, bh = w * iw, h * ih
            obj = ET.SubElement(ann, "object")
            name = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            ET.SubElement(obj, "name").text = name
            bnd = ET.SubElement(obj, "bndbox")
            ET.SubElement(bnd, "xmin").text = str(max(0, int(round(cx * iw - bw / 2))))
            ET.SubElement(bnd, "ymin").text = str(max(0, int(round(cy * ih - bh / 2))))
            ET.SubElement(bnd, "xmax").text = str(min(iw, int(round(cx * iw + bw / 2))))
            ET.SubElement(bnd, "ymax").text = str(min(ih, int(round(cy * ih + bh / 2))))
        ET.ElementTree(ann).write(out_dir / f"{img_path.stem}.xml", encoding="unicode")
        written += 1
    return written


def yolo_to_cvat_csv(
    images_dir: str | Path,
    labels_dir: str | Path | None,
    out_csv: str | Path,
    class_names: list[str],
    *,
    template_csv: str | Path | None = None,
) -> int:
    """Convert a YOLO dataset to a flat CVAT CSV export (the GT counterpart of
    :func:`predictions_to_cvat_csv`).

    One row per box, **rectangles only**. Bookkeeping/extra columns (task/job
    ids, image_path, a CVAT link, …) are joined from ``template_csv`` by
    ``image_name`` when given. Images with no boxes contribute no rows.

    Args:
        images_dir:   Directory of images (used for the per-image pixel size).
        labels_dir:   Directory of YOLO ``.txt`` labels (inferred if ``None``).
        out_csv:      Destination ``.csv`` path.
        class_names:  Ordered class names (index = YOLO class id).
        template_csv: Optional CVAT CSV supplying per-image metadata/extra columns.

    Returns:
        Number of instance rows written.
    """
    extra_cols, by_name = _load_cvat_template(template_csv)
    columns = list(CVAT_CSV_COLUMNS) + extra_cols

    all_rows: list[dict[str, str]] = []
    for img_path, label_path in _iter_yolo_pairs(Path(images_dir), labels_dir):
        boxes = _read_yolo_boxes(label_path)
        if not boxes:
            continue
        iw, ih = _image_size(img_path)
        xyxy = [
            (cx * iw - w * iw / 2, cy * ih - h * ih / 2, cx * iw + w * iw / 2, cy * ih + h * ih / 2)
            for _, cx, cy, w, h in boxes
        ]
        labels = [class_names[cid] if 0 <= cid < len(class_names) else str(cid) for cid, *_ in boxes]
        meta = by_name.get(img_path.name, {})
        all_rows.extend(_cvat_rows_for_image(img_path.name, iw, ih, xyxy, labels, meta, columns))

    _write_cvat_csv(all_rows, out_csv, columns)
    return len(all_rows)


# ---------------------------------------------------------------------------
# Ultralytics predictions → CVAT CSV (reverse of cvat_csv_to_yolo)
# ---------------------------------------------------------------------------

def _load_cvat_template(
    template_csv: str | Path | None,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Read a template CVAT CSV → (extra columns, image_name → first row).

    Extra columns are those beyond :data:`CVAT_CSV_COLUMNS` (e.g. a CVAT link).
    The per-image row supplies bookkeeping (task/job/image ids, image_path, …)
    that a model cannot know; the first row seen for an image carries it.
    """
    if template_csv is None:
        return [], {}
    header, rows = _read_cvat_csv(template_csv)
    extra = [c for c in header if c not in CVAT_CSV_COLUMNS]
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get("image_name") or row.get("image_path")
        if name:
            by_name.setdefault(Path(name).name, row)
    return extra, by_name


def _cvat_rows_for_image(
    image_name: str,
    width: int,
    height: int,
    boxes_xyxy,
    labels: list[str],
    meta: dict[str, str],
    columns: list[str],
) -> list[dict[str, str]]:
    """Build one CVAT CSV row per box (rectangles only) for a single image.

    Bookkeeping/extra columns are seeded from ``meta`` (the template row); the
    per-detection fields (size, label, shape, bbox corners) are then overwritten.
    """
    base = {c: meta.get(c, "") for c in columns}
    base["image_name"] = image_name
    base["image_width"] = str(int(width))
    base["image_height"] = str(int(height))
    base["instance_shape"] = "rectangle"
    base["instance_points"] = ""

    rows: list[dict[str, str]] = []
    for (x1, y1, x2, y2), label in zip(boxes_xyxy, labels):
        row = dict(base)
        row["instance_label"] = label
        row["bbox_x_tl"] = f"{float(x1):.2f}"
        row["bbox_y_tl"] = f"{float(y1):.2f}"
        row["bbox_x_br"] = f"{float(x2):.2f}"
        row["bbox_y_br"] = f"{float(y2):.2f}"
        rows.append(row)
    return rows


def _write_cvat_csv(rows: list[dict[str, str]], out_csv: str | Path, columns: list[str]) -> None:
    """Write rows to a CVAT CSV (utf-8-sig, the encoding _read_cvat_csv expects)."""
    import csv

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def predictions_to_cvat_csv(
    model,
    images_dir: str | Path,
    out_csv: str | Path,
    *,
    template_csv: str | Path | None = None,
    class_names: list[str] | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str | None = None,
    extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
) -> int:
    """Run an Ultralytics detector over a folder and write a CVAT CSV export.

    The reverse of :func:`cvat_csv_to_yolo`: model detections become per-instance
    rows in the flat CVAT CSV schema (:data:`CVAT_CSV_COLUMNS`), ready to upload
    back into CVAT for human correction. **Rectangles only** —
    ``instance_shape`` is always ``"rectangle"`` and ``instance_points`` empty.

    Bookkeeping columns a model cannot produce (``image_id``, ``job_id``,
    ``task_id``, ``task_name``, ``task_assignee``, ``image_path``) — plus any
    extra columns such as a CVAT link — are joined from ``template_csv`` by
    ``image_name``. Without a template they stay blank. Images with no detections
    contribute no rows.

    Args:
        model:        Ultralytics ``YOLO`` instance or path to a ``.pt`` file.
        images_dir:   Directory of images (searched recursively).
        out_csv:      Destination ``.csv`` path.
        template_csv: Optional CVAT CSV supplying per-image metadata/extra columns.
        class_names:  Class names by index; falls back to the model's ``names``.
        conf:         Confidence threshold for kept detections.
        iou:          NMS IoU threshold.
        imgsz:        Inference image size.
        device:       Optional device override (e.g. ``"cpu"``, ``"0"``).
        extensions:   Image extensions to include.

    Returns:
        Number of instance rows written.
    """
    import numpy as np

    from cv_lib.data.autolabel import _load_model

    model = _load_model(model)
    if class_names is None:
        names = getattr(model, "names", None)
        class_names = list(names.values()) if names else []

    extra_cols, by_name = _load_cvat_template(template_csv)
    columns = list(CVAT_CSV_COLUMNS) + extra_cols

    images_dir = Path(images_dir)
    image_files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in extensions)

    predict_kwargs: dict = {"conf": conf, "iou": iou, "imgsz": imgsz, "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device

    all_rows: list[dict[str, str]] = []
    for img_path in image_files:
        result = model.predict(source=str(img_path), **predict_kwargs)[0]
        boxes = result.boxes
        if boxes is None or not len(boxes):
            continue
        h, w = result.orig_shape
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        cls = cls.astype(int)
        labels = [class_names[c] if 0 <= c < len(class_names) else str(c) for c in cls]
        meta = by_name.get(img_path.name, {})
        all_rows.extend(_cvat_rows_for_image(img_path.name, w, h, xyxy, labels, meta, columns))

    _write_cvat_csv(all_rows, out_csv, columns)
    return len(all_rows)
