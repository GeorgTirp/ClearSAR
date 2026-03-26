#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an augmented ClearSAR dataset in a new output root. "
            "The output contains a full copy of the original dataset plus "
            "train-set augmentations (one random flip + one random crop per image)."
        )
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        default=Path("src/data/ClearSAR/data"),
        help="Source dataset root with images/ and annotations/.",
    )
    parser.add_argument(
        "--output-data-root",
        type=Path,
        default=Path("src/data/ClearSAR/augmented_data"),
        help="Output dataset root to create (full source copy + augmented train set).",
    )
    parser.add_argument(
        "--train-ann",
        type=Path,
        default=None,
        help="Optional source COCO train annotation file. Defaults to source-data-root/annotations/instances_train.json.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Optional source train images root. Defaults to source-data-root/images/train.",
    )
    parser.add_argument(
        "--output-ann",
        type=Path,
        default=None,
        help="Optional output train annotation file. Defaults to output-data-root/annotations/instances_train.json.",
    )
    parser.add_argument(
        "--output-images-dir",
        type=Path,
        default=None,
        help="Optional output train images dir for augmented images. Defaults to output-data-root/images/train.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=0.8,
        help="Random crop size ratio relative to original width/height.",
    )
    parser.add_argument(
        "--min-visible-frac",
        type=float,
        default=0.2,
        help="Minimum visible fraction of original bbox area to keep a cropped bbox.",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=4.0,
        help="Minimum bbox area in pixels after crop to keep annotation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic flip/crop randomness.",
    )
    parser.add_argument(
        "--no-include-original",
        action="store_true",
        help="If set, output train annotations contain only augmented samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-data-root if it already exists.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first missing source image; otherwise skip missing images.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def resolve_train_ann(source_data_root: Path, train_ann: Optional[Path]) -> Path:
    if train_ann is not None:
        if not train_ann.exists():
            raise FileNotFoundError(f"Training annotation not found: {train_ann}")
        return train_ann
    candidates = [
        source_data_root / "annotations" / "instances_train.json",
        source_data_root / "annotations" / "train.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find source train annotation. Looked for: "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_source_train_images_root(source_data_root: Path, images_root: Optional[Path]) -> Path:
    root = images_root if images_root is not None else source_data_root / "images" / "train"
    if not root.exists():
        raise FileNotFoundError(f"Source train images root not found: {root}")
    return root


def copy_source_dataset(source_data_root: Path, output_data_root: Path, overwrite: bool) -> None:
    source_resolved = source_data_root.resolve()
    output_resolved = output_data_root.resolve()
    if source_resolved == output_resolved:
        raise ValueError("source-data-root and output-data-root must be different paths.")
    if not source_data_root.exists():
        raise FileNotFoundError(f"Source dataset root not found: {source_data_root}")

    if output_data_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset root already exists: {output_data_root}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(output_data_root)

    shutil.copytree(source_data_root, output_data_root)


def resolve_image_path(file_name: str, source_train_images_root: Path) -> Optional[Path]:
    candidates = [
        source_train_images_root / file_name,
        source_train_images_root.parent / file_name,
        source_train_images_root.parent / "train" / file_name,
        source_train_images_root.parent / "val" / file_name,
        source_train_images_root.parent / "test" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def flip_bbox_xywh(bbox: List[float], width: int, height: int, mode: str) -> List[float]:
    x, y, w, h = [float(v) for v in bbox]
    if mode == "h":
        return [float(width - x - w), y, w, h]
    if mode == "v":
        return [x, float(height - y - h), w, h]
    raise ValueError(f"Unsupported flip mode: {mode}")


def crop_bbox_xywh(
    bbox: List[float],
    crop_left: int,
    crop_top: int,
    crop_width: int,
    crop_height: int,
    min_visible_frac: float,
    min_box_area: float,
) -> Optional[List[float]]:
    x, y, w, h = [float(v) for v in bbox]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    cx1 = max(x1, float(crop_left))
    cy1 = max(y1, float(crop_top))
    cx2 = min(x2, float(crop_left + crop_width))
    cy2 = min(y2, float(crop_top + crop_height))

    nw = max(0.0, cx2 - cx1)
    nh = max(0.0, cy2 - cy1)
    new_area = nw * nh
    old_area = max(1e-6, w * h)
    visible_frac = new_area / old_area

    if new_area < min_box_area or visible_frac < min_visible_frac:
        return None

    return [cx1 - float(crop_left), cy1 - float(crop_top), nw, nh]


def clone_annotation(
    base_ann: Dict[str, Any],
    ann_id: int,
    image_id: int,
    bbox_xywh: List[float],
) -> Dict[str, Any]:
    out = dict(base_ann)
    out["id"] = ann_id
    out["image_id"] = image_id
    out["bbox"] = [float(v) for v in bbox_xywh]
    out["area"] = float(bbox_xywh[2] * bbox_xywh[3])
    out["iscrowd"] = int(base_ann.get("iscrowd", 0))
    # Keep detection fields valid after geometric transforms.
    out["segmentation"] = []
    return out


def main() -> None:
    args = parse_args()

    if not (0.0 < args.crop_scale <= 1.0):
        raise ValueError(f"--crop-scale must be in (0, 1], got {args.crop_scale}")
    if not (0.0 <= args.min_visible_frac <= 1.0):
        raise ValueError(f"--min-visible-frac must be in [0, 1], got {args.min_visible_frac}")
    if args.min_box_area < 0.0:
        raise ValueError(f"--min-box-area must be >= 0, got {args.min_box_area}")

    source_train_ann = resolve_train_ann(args.source_data_root, args.train_ann)
    source_train_images_root = resolve_source_train_images_root(args.source_data_root, args.images_root)

    output_ann = (
        args.output_ann
        if args.output_ann is not None
        else args.output_data_root / "annotations" / "instances_train.json"
    )
    output_images_dir = (
        args.output_images_dir
        if args.output_images_dir is not None
        else args.output_data_root / "images" / "train"
    )

    copy_source_dataset(
        source_data_root=args.source_data_root,
        output_data_root=args.output_data_root,
        overwrite=args.overwrite,
    )
    output_images_dir.mkdir(parents=True, exist_ok=True)

    coco = read_json(source_train_ann)
    if not all(k in coco for k in ("images", "annotations", "categories")):
        raise ValueError("Input annotation must contain COCO keys: images, annotations, categories.")

    images: List[Dict[str, Any]] = sorted(coco["images"], key=lambda item: int(item["id"]))
    annotations: List[Dict[str, Any]] = coco["annotations"]
    ann_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        ann_by_image[int(ann["image_id"])].append(ann)

    include_original = not args.no_include_original
    output_images: List[Dict[str, Any]] = [dict(img) for img in images] if include_original else []
    output_annotations: List[Dict[str, Any]] = [dict(ann) for ann in annotations] if include_original else []

    next_image_id = max((int(img["id"]) for img in output_images), default=-1) + 1
    next_ann_id = max((int(ann["id"]) for ann in output_annotations), default=-1) + 1

    rng = random.Random(args.seed)
    missing_images = 0
    generated_images = 0
    generated_annotations = 0
    dropped_crop_boxes = 0

    for img_entry in images:
        src_image_id = int(img_entry["id"])
        src_file_name = str(img_entry["file_name"])
        src_path = resolve_image_path(src_file_name, source_train_images_root)
        if src_path is None:
            missing_images += 1
            message = f"Missing source image for file_name='{src_file_name}' (images-root={source_train_images_root})"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[skip] {message}")
            continue

        with Image.open(src_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            anns = ann_by_image.get(src_image_id, [])

            image_stem = Path(src_file_name).stem
            image_ext = Path(src_file_name).suffix or ".png"

            # Flip augmentation: choose horizontal or vertical with 50/50 probability.
            flip_mode = "h" if rng.random() < 0.5 else "v"
            flip_tag = "flip_h" if flip_mode == "h" else "flip_v"
            flip_out_name = f"{image_stem}_{src_image_id}__{flip_tag}{image_ext}"
            flip_out_path = output_images_dir / flip_out_name
            if flip_mode == "h":
                flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                flip_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            flip_image.save(flip_out_path)

            flip_image_id = next_image_id
            next_image_id += 1
            flip_image_entry = dict(img_entry)
            flip_image_entry["id"] = flip_image_id
            flip_image_entry["file_name"] = flip_out_name
            flip_image_entry["width"] = width
            flip_image_entry["height"] = height
            output_images.append(flip_image_entry)
            generated_images += 1

            for ann in anns:
                bbox = flip_bbox_xywh(ann["bbox"], width=width, height=height, mode=flip_mode)
                output_annotations.append(
                    clone_annotation(
                        base_ann=ann,
                        ann_id=next_ann_id,
                        image_id=flip_image_id,
                        bbox_xywh=bbox,
                    )
                )
                next_ann_id += 1
                generated_annotations += 1

            # Crop augmentation: random crop position with fixed crop-scale.
            crop_width = max(1, min(width, int(round(width * args.crop_scale))))
            crop_height = max(1, min(height, int(round(height * args.crop_scale))))
            max_left = max(0, width - crop_width)
            max_top = max(0, height - crop_height)
            crop_left = rng.randint(0, max_left)
            crop_top = rng.randint(0, max_top)

            crop_out_name = f"{image_stem}_{src_image_id}__crop{image_ext}"
            crop_out_path = output_images_dir / crop_out_name
            crop_image = image.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
            crop_image.save(crop_out_path)

            crop_image_id = next_image_id
            next_image_id += 1
            crop_image_entry = dict(img_entry)
            crop_image_entry["id"] = crop_image_id
            crop_image_entry["file_name"] = crop_out_name
            crop_image_entry["width"] = crop_width
            crop_image_entry["height"] = crop_height
            output_images.append(crop_image_entry)
            generated_images += 1

            for ann in anns:
                cropped_bbox = crop_bbox_xywh(
                    bbox=ann["bbox"],
                    crop_left=crop_left,
                    crop_top=crop_top,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    min_visible_frac=args.min_visible_frac,
                    min_box_area=args.min_box_area,
                )
                if cropped_bbox is None:
                    dropped_crop_boxes += 1
                    continue
                output_annotations.append(
                    clone_annotation(
                        base_ann=ann,
                        ann_id=next_ann_id,
                        image_id=crop_image_id,
                        bbox_xywh=cropped_bbox,
                    )
                )
                next_ann_id += 1
                generated_annotations += 1

    out_payload: Dict[str, Any] = {k: v for k, v in coco.items() if k not in {"images", "annotations"}}
    out_payload["images"] = output_images
    out_payload["annotations"] = output_annotations

    write_json(output_ann, out_payload)

    print(
        "Augmentation complete: "
        f"source_root={args.source_data_root} "
        f"output_root={args.output_data_root} "
        f"base_images={len(images)} "
        f"base_annotations={len(annotations)} "
        f"generated_images={generated_images} "
        f"generated_annotations={generated_annotations} "
        f"dropped_crop_boxes={dropped_crop_boxes} "
        f"missing_images={missing_images} "
        f"-> {output_ann}"
    )


if __name__ == "__main__":
    main()
