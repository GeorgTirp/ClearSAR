#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SARDet into a clean COCO-style detection layout."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("src/data/SARDet_100K"),
        help="Source SARDet root containing JPEGImages/ and Annotations/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("src/data/SARDet_COCO"),
        help="Output root where images/ and annotations/ will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Splits to export.",
    )
    parser.add_argument(
        "--source-bbox-format",
        choices=["xywh", "xyxy"],
        default="xywh",
        help="Bounding box format in source annotations.",
    )
    parser.add_argument(
        "--class-id-start",
        type=int,
        default=0,
        help="Starting class ID for contiguous remap.",
    )
    parser.add_argument(
        "--image-transfer",
        choices=["symlink", "hardlink", "copy", "none"],
        default="symlink",
        help="How to place images into output images/<split>/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and recreate output split directories/files if present.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first missing image or invalid annotation.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "none":
        return
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported image-transfer mode: {mode}")


def normalize_bbox(bbox: List[float], src_format: str) -> Tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox length 4, got {len(bbox)}")

    if src_format == "xywh":
        x, y, w, h = bbox
    else:
        x1, y1, x2, y2 = bbox
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
    return float(x), float(y), float(w), float(h)


def remap_categories(categories: List[Dict], class_id_start: int) -> Tuple[List[Dict], Dict[int, int]]:
    ordered = sorted(categories, key=lambda c: c["id"])
    id_map: Dict[int, int] = {}
    remapped: List[Dict] = []
    for idx, cat in enumerate(ordered):
        old_id = int(cat["id"])
        new_id = class_id_start + idx
        id_map[old_id] = new_id
        new_cat = dict(cat)
        new_cat["id"] = new_id
        remapped.append(new_cat)
    return remapped, id_map


def contiguous_ids(ids: List[int]) -> bool:
    if not ids:
        return True
    min_id, max_id = min(ids), max(ids)
    return sorted(set(ids)) == list(range(min_id, max_id + 1))


def export_split(
    split: str,
    input_root: Path,
    output_root: Path,
    source_bbox_format: str,
    class_id_start: int,
    image_transfer: str,
    overwrite: bool,
    strict: bool,
) -> None:
    src_ann = input_root / "Annotations" / f"{split}.json"
    if not src_ann.exists():
        raise FileNotFoundError(f"Missing annotation file for split '{split}': {src_ann}")

    src = read_json(src_ann)
    if not all(k in src for k in ("images", "annotations", "categories")):
        raise ValueError(f"{src_ann} is missing COCO keys (images/annotations/categories).")

    dst_images_root = output_root / "images" / split
    dst_ann_file = output_root / "annotations" / f"{split}.json"

    ensure_clean_dir(dst_images_root, overwrite=overwrite)
    if dst_ann_file.exists() and overwrite:
        dst_ann_file.unlink()

    categories_out, cat_id_map = remap_categories(src["categories"], class_id_start=class_id_start)
    new_images: List[Dict] = []
    image_id_map: Dict[int, int] = {}
    missing_images = 0

    for new_image_id, image in enumerate(src["images"]):
        old_image_id = int(image["id"])
        image_id_map[old_image_id] = new_image_id

        file_name = image["file_name"]
        src_path = input_root / "JPEGImages" / split / file_name
        dst_rel = f"{split}/{file_name}"
        dst_path = output_root / "images" / dst_rel

        if not src_path.exists():
            missing_images += 1
            if strict:
                raise FileNotFoundError(f"Missing source image: {src_path}")
            continue

        link_or_copy(src_path, dst_path, image_transfer)

        new_image = dict(image)
        new_image["id"] = new_image_id
        new_image["file_name"] = dst_rel
        new_images.append(new_image)

    new_annotations: List[Dict] = []
    invalid_boxes = 0

    for new_ann_id, ann in enumerate(src["annotations"]):
        old_image_id = int(ann["image_id"])
        if old_image_id not in image_id_map:
            continue
        old_cat = int(ann["category_id"])
        if old_cat not in cat_id_map:
            raise ValueError(f"Annotation references unknown category ID {old_cat}.")

        try:
            x, y, w, h = normalize_bbox(ann["bbox"], source_bbox_format)
        except Exception:
            invalid_boxes += 1
            if strict:
                raise
            continue

        if w < 0 or h < 0:
            invalid_boxes += 1
            if strict:
                raise ValueError(f"Negative bbox size found in annotation ID {ann.get('id')}.")
            continue

        new_ann = dict(ann)
        new_ann["id"] = new_ann_id
        new_ann["image_id"] = image_id_map[old_image_id]
        new_ann["category_id"] = cat_id_map[old_cat]
        new_ann["bbox"] = [x, y, w, h]  # COCO xywh
        new_ann["area"] = float(new_ann.get("area", w * h)) if new_ann.get("area") is not None else float(w * h)
        new_ann["iscrowd"] = int(new_ann.get("iscrowd", 0))
        if "segmentation" not in new_ann:
            new_ann["segmentation"] = []
        new_annotations.append(new_ann)

    out = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories_out,
    }
    if "info" in src:
        out["info"] = src["info"]
    if "licenses" in src:
        out["licenses"] = src["licenses"]

    cat_ids = [c["id"] for c in categories_out]
    ann_cat_ids = [a["category_id"] for a in new_annotations]
    if not contiguous_ids(cat_ids):
        raise ValueError(f"Category IDs are not contiguous in {split}: {sorted(set(cat_ids))}")
    if ann_cat_ids and not set(ann_cat_ids).issubset(set(cat_ids)):
        raise ValueError(f"Annotation category IDs not subset of category IDs in {split}.")

    for ann in new_annotations:
        bbox = ann["bbox"]
        if len(bbox) != 4:
            raise ValueError(f"Non-xywh bbox in {split}, annotation {ann['id']}: {bbox}")

    write_json(dst_ann_file, out)

    print(
        f"[{split}] images={len(new_images)} annotations={len(new_annotations)} "
        f"categories={len(categories_out)} missing_images={missing_images} invalid_boxes={invalid_boxes} "
        f"-> {dst_ann_file}"
    )


def main() -> None:
    args = parse_args()
    for split in args.splits:
        export_split(
            split=split,
            input_root=args.input_root,
            output_root=args.output_root,
            source_bbox_format=args.source_bbox_format,
            class_id_start=args.class_id_start,
            image_transfer=args.image_transfer,
            overwrite=args.overwrite,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
