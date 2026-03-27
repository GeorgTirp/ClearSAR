#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import DINOv2DINOModel
from src.model.backbone import DINOV2_BACKBONE_CHOICES, DINOV2_PATCH_SIZE
from src.optim import MultiOptimizer, build_kl_shampoo_optimizer, build_muon_optimizer
from scripts.augmentation import augment_train_dataset


PATH_CONFIG_KEYS = {
    "data_root",
    "train_ann",
    "val_ann",
    "test_ann",
    "images_root",
    "test_images_root",
    "augmentation_output_root",
    "output_dir",
    "resume_checkpoint",
}


OptimizerLike = torch.optim.Optimizer | MultiOptimizer

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
DEFAULT_SPECTRAL_FUSE_LEVELS = (1, 2, 3)
DEFAULT_AUGMENTED_FILENAME_MARKERS = ("__flip_h", "__flip_v", "__crop", "__aug")


def load_config_file(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "YAML config requires PyYAML. Install it with `pip install pyyaml`."
            ) from exc
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
    else:
        raise ValueError(f"Only JSON or YAML config is supported, got: {config_path}")

    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON/YAML object, got: {type(payload).__name__}")

    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        dest = key.replace("-", "_")
        if dest in PATH_CONFIG_KEYS and value is not None:
            normalized[dest] = Path(value)
        else:
            normalized[dest] = value
    return normalized


def create_train_val_split_annotations(
    source_ann: Path,
    val_ratio: float,
    split_seed: int,
    val_originals_only: bool = False,
    augmented_filename_markers: Sequence[str] = DEFAULT_AUGMENTED_FILENAME_MARKERS,
) -> Tuple[Path, Path]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"train/val split ratio must be in (0, 1), got {val_ratio}.")

    payload = json.loads(source_ann.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if not images:
        raise ValueError(f"No images found in annotation file: {source_ann}")

    image_ids = [int(image["id"]) for image in images]
    if val_originals_only:
        candidate_val_ids = [
            int(image["id"])
            for image in images
            if not is_augmented_image_entry(image, augmented_filename_markers)
        ]
        if not candidate_val_ids:
            raise ValueError(
                "Validation split requested originals-only images, but no original images were "
                f"found in {source_ann}. Disable --val-originals-only or check augmented filename markers."
            )
        shuffled_ids = candidate_val_ids[:]
    else:
        shuffled_ids = image_ids[:]
    rng = random.Random(split_seed)
    rng.shuffle(shuffled_ids)

    val_count = max(1, int(round(len(shuffled_ids) * val_ratio)))
    if val_count >= len(shuffled_ids):
        val_count = len(shuffled_ids) - 1
    if val_count <= 0:
        raise ValueError(
            f"Could not create non-empty train/val split from {len(shuffled_ids)} images. "
            f"Choose a different ratio than {val_ratio}."
        )

    val_ids = set(shuffled_ids[:val_count])
    train_ids = set(image_ids) - val_ids
    if not train_ids:
        raise ValueError("Auto split produced an empty training set.")

    def build_split(selected_ids: set[int]) -> Dict[str, Any]:
        split_images = [image for image in images if int(image["id"]) in selected_ids]
        split_anns = [ann for ann in annotations if int(ann["image_id"]) in selected_ids]
        split_payload = {k: v for k, v in payload.items() if k not in {"images", "annotations"}}
        split_payload["images"] = split_images
        split_payload["annotations"] = split_anns
        return split_payload

    train_pct = int(round((1.0 - val_ratio) * 100.0))
    val_pct = 100 - train_pct
    split_prefix = f"{source_ann.stem}_split_{train_pct}_{val_pct}_seed{split_seed}"
    train_split_path = source_ann.parent / f"{split_prefix}_train.json"
    val_split_path = source_ann.parent / f"{split_prefix}_val.json"

    train_split_path.write_text(json.dumps(build_split(train_ids), indent=2), encoding="utf-8")
    val_split_path.write_text(json.dumps(build_split(val_ids), indent=2), encoding="utf-8")

    return train_split_path, val_split_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly fine-tune ClearSAR with pretrained DINOv2 + Deformable DINO."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON or YAML config file. CLI args override config values.",
    )

    parser.add_argument("--data-root", type=Path, default=Path("src/data/ClearSAR/data"))
    parser.add_argument("--train-ann", type=Path, default=None)
    parser.add_argument("--val-ann", type=Path, default=None)
    parser.add_argument("--test-ann", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--test-images-root", type=Path, default=None)
    parser.add_argument(
        "--train-val-split-ratio",
        type=float,
        default=0.2,
        help="Validation ratio used when auto-splitting train annotations (default: 0.2 for 80/20).",
    )
    parser.add_argument(
        "--train-val-split-seed",
        type=int,
        default=42,
        help="Seed used for deterministic train/val split generation.",
    )
    parser.add_argument(
        "--no-auto-train-val-split",
        action="store_true",
        help="Disable automatic train/val split generation when no val annotation file is found.",
    )
    parser.add_argument(
        "--val-originals-only",
        action="store_true",
        help=(
            "Exclude augmented samples from validation/evaluation datasets. "
            "When auto-splitting train annotations, only original images can be chosen for validation."
        ),
    )
    parser.add_argument(
        "--no-val-originals-only",
        action="store_false",
        dest="val_originals_only",
        help="Allow augmented samples in validation/evaluation datasets.",
    )
    parser.set_defaults(val_originals_only=True)
    parser.add_argument(
        "--augmented-filename-markers",
        type=str,
        default=",".join(DEFAULT_AUGMENTED_FILENAME_MARKERS),
        help=(
            "Comma-separated substrings used to detect augmented images by file_name "
            "(used when image metadata does not include is_augmented)."
        ),
    )
    parser.add_argument(
        "--augment-train-after-split",
        action="store_true",
        help=(
            "Run dataset augmentation automatically before training (after auto train/val split if it is used). "
            "The augmented train annotation replaces --train-ann for this run."
        ),
    )
    parser.add_argument(
        "--augmentation-output-root",
        type=Path,
        default=None,
        help=(
            "Output dataset root used by --augment-train-after-split. "
            "Defaults to <output-dir>/runtime_augmented_data."
        ),
    )
    parser.add_argument(
        "--augmentation-crop-scale",
        type=float,
        default=0.8,
        help="Crop scale forwarded to runtime augmentation.",
    )
    parser.add_argument(
        "--augmentation-min-visible-frac",
        type=float,
        default=0.2,
        help="Minimum visible bbox fraction forwarded to runtime augmentation.",
    )
    parser.add_argument(
        "--augmentation-min-box-area",
        type=float,
        default=4.0,
        help="Minimum cropped bbox area forwarded to runtime augmentation.",
    )
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=42,
        help="Seed forwarded to runtime augmentation.",
    )
    parser.add_argument(
        "--no-augmentation-include-original",
        action="store_false",
        dest="augmentation_include_original",
        help="Use only generated augmentations in the runtime-augmented train annotation.",
    )
    parser.set_defaults(augmentation_include_original=True)
    parser.add_argument(
        "--augmentation-overwrite",
        action="store_true",
        help="Overwrite --augmentation-output-root if it already exists.",
    )
    parser.add_argument(
        "--augmentation-strict",
        action="store_true",
        help="Fail runtime augmentation on first missing source image.",
    )

    parser.add_argument(
        "--backbone-name",
        type=str,
        default="vit_base_patch14_dinov2",
        choices=list(DINOV2_BACKBONE_CHOICES),
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=None,
        help="Optional local checkpoint path for the selected DINOv2 backbone.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Path to a training checkpoint (.pth) to resume from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/clearsar_finetune"),
    )

    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-score-threshold", type=float, default=0.05)
    parser.add_argument("--eval-topk", type=int, default=300)

    parser.add_argument("--phase-a-epochs", type=int, default=5)
    parser.add_argument("--phase-b-epochs", type=int, default=45)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "kl_shampoo", "muon"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.1)
    parser.add_argument("--soap-betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--soap-shampoo-beta", type=float, default=0.95)
    parser.add_argument("--soap-eps", type=float, default=1e-8)
    parser.add_argument(
        "--soap-weight-decay-method",
        type=str,
        default="decoupled",
        choices=["decoupled", "independent", "l2"],
    )
    parser.add_argument("--soap-nesterov", action="store_true")
    parser.add_argument("--soap-precondition-frequency", type=int, default=1)
    parser.add_argument("--soap-adam-warmup-steps", type=int, default=0)
    parser.add_argument("--soap-correct-bias", action="store_true")
    parser.add_argument(
        "--soap-fp32-matmul-prec",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
    )
    parser.add_argument("--soap-use-eigh", action="store_true")
    parser.add_argument(
        "--soap-qr-fp32-matmul-prec",
        type=str,
        default="high",
        choices=["highest", "high", "medium"],
    )
    parser.add_argument("--soap-use-adaptive-criteria", action="store_true")
    parser.add_argument("--soap-adaptive-update-tolerance", type=float, default=1e-7)
    parser.add_argument("--soap-power-iter-steps", type=int, default=1)
    parser.add_argument("--soap-max-update-rms", type=float, default=0.0)
    parser.add_argument("--soap-no-kl-shampoo", action="store_true")
    parser.add_argument(
        "--soap-correct-shampoo-beta-bias",
        type=str,
        default="none",
        choices=["none", "true", "false"],
    )
    parser.add_argument("--muon-lr-mult", type=float, default=20.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-adam-betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--muon-adam-eps", type=float, default=1e-8)

    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-feature-levels", type=int, default=4)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=6)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--enc-n-points", type=int, default=4)
    parser.add_argument("--dec-n-points", type=int, default=4)
    parser.add_argument("--no-two-stage", action="store_true")
    parser.add_argument("--no-box-refine", action="store_true")
    parser.add_argument("--no-object-class", action="store_false", dest="include_no_object")
    parser.set_defaults(include_no_object=True)
    parser.add_argument("--no-spectral-prior", action="store_true")
    parser.add_argument(
        "--spectral-blur-kernel-size",
        type=int,
        default=31,
        help="Legacy no-op for backward config compatibility.",
    )
    parser.add_argument(
        "--spectral-blur-sigma",
        type=float,
        default=6.0,
        help="Legacy no-op for backward config compatibility.",
    )
    parser.add_argument("--spectral-branch-channels", type=int, default=64)
    parser.add_argument(
        "--spectral-patch-size",
        type=int,
        default=8,
        help="Patch size used for patch-wise DCT prior extraction.",
    )
    parser.add_argument(
        "--spectral-dct-keep-size",
        type=int,
        default=4,
        help="Keep a top-left KxK DCT block per patch (read in zig-zag order).",
    )
    parser.add_argument(
        "--spectral-drop-dc",
        action="store_true",
        help="Drop the patch DC coefficient before the spectral MLP encoder.",
    )
    parser.add_argument(
        "--spectral-fuse-level-indices",
        type=str,
        default=",".join(str(v) for v in DEFAULT_SPECTRAL_FUSE_LEVELS),
        help="Comma-separated feature level indices used for spectral prior fusion, e.g. '1,2,3'.",
    )

    parser.add_argument("--cls-loss-coef", type=float, default=1.0)
    parser.add_argument("--bbox-loss-coef", type=float, default=5.0)
    parser.add_argument("--giou-loss-coef", type=float, default=2.0)
    parser.add_argument("--eos-coef", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--wandb-enable", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default="clearsar-finetune")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default="train")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--wandb-use-sweep-config",
        action="store_true",
        help="Apply matching keys from wandb.config to training args (for sweeps).",
    )
    parser.add_argument("--wandb-watch", action="store_true", help="Enable gradient/parameter watching.")
    parser.add_argument("--wandb-log-freq", type=int, default=100)

    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    bootstrap_args, _ = parser.parse_known_args()
    config_values = load_config_file(bootstrap_args.config)

    if config_values:
        valid_keys = {action.dest for action in parser._actions}
        unknown = sorted(set(config_values.keys()) - valid_keys)
        if unknown:
            parser.error(f"Unknown config keys: {unknown}")
        parser.set_defaults(**config_values)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


class ClearSARCocoDataset(Dataset):
    def __init__(
        self,
        images_root: Path,
        ann_file: Path,
        image_size: int,
        originals_only: bool = False,
        augmented_filename_markers: Sequence[str] = DEFAULT_AUGMENTED_FILENAME_MARKERS,
    ) -> None:
        self.images_root = images_root
        self.ann_file = ann_file
        self.image_size = image_size
        self.image_search_subdirs = self._infer_image_search_subdirs(ann_file)
        self.augmented_filename_markers = tuple(augmented_filename_markers)

        data = json.loads(ann_file.read_text(encoding="utf-8"))
        all_images = sorted(data["images"], key=lambda x: x["id"])
        all_annotations = data["annotations"]
        if originals_only:
            kept_images = [
                image
                for image in all_images
                if not is_augmented_image_entry(image, self.augmented_filename_markers)
            ]
            if not kept_images:
                raise ValueError(
                    "Validation originals-only filtering removed all images from "
                    f"{ann_file}. Disable --val-originals-only or adjust --augmented-filename-markers."
                )
            kept_ids = {int(image["id"]) for image in kept_images}
            kept_annotations = [ann for ann in all_annotations if int(ann["image_id"]) in kept_ids]
            removed_images = len(all_images) - len(kept_images)
            removed_annotations = len(all_annotations) - len(kept_annotations)
            if removed_images > 0:
                print(
                    "[dataset] originals-only filter: "
                    f"ann={ann_file.name} removed_images={removed_images} "
                    f"removed_annotations={removed_annotations} "
                    f"markers={self.augmented_filename_markers}"
                )
            self.images = kept_images
            annotations = kept_annotations
        else:
            self.images = all_images
            annotations = all_annotations
        self.categories = sorted(data["categories"], key=lambda x: x["id"])
        self.image_to_anns: Dict[int, List[Dict]] = {}
        for ann in annotations:
            image_id = int(ann["image_id"])
            self.image_to_anns.setdefault(image_id, []).append(ann)

    @staticmethod
    def _infer_image_search_subdirs(ann_file: Path) -> List[str]:
        name = ann_file.stem.lower()
        if "train" in name:
            return ["train", "val", "test"]
        if "val" in name or "test" in name:
            return ["val", "test", "train"]
        return ["train", "val", "test"]

    def _resolve_image_path(self, file_name: str) -> Path:
        candidates = [self.images_root / file_name] + [
            self.images_root / subdir / file_name for subdir in self.image_search_subdirs
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        checked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing image '{file_name}'. Checked: {checked}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_info = self.images[idx]
        img_path = self._resolve_image_path(image_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        if self.image_size > 0:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            out_w = out_h = self.image_size
        else:
            out_w, out_h = orig_w, orig_h

        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD

        anns = self.image_to_anns.get(int(image_info["id"]), [])
        boxes = []
        labels = []
        sx = out_w / float(orig_w)
        sy = out_h / float(orig_h)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x *= sx
            y *= sy
            w *= sx
            h *= sy
            cx = (x + 0.5 * w) / out_w
            cy = (y + 0.5 * h) / out_h
            nw = w / out_w
            nh = h / out_h
            boxes.append([cx, cy, nw, nh])
            labels.append(int(ann["category_id"]))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(int(image_info["id"]), dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "size": torch.tensor([out_h, out_w], dtype=torch.long),
        }
        return img_tensor, target


def collate_fn(batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)


def parse_augmented_filename_markers(raw_value: Any) -> Tuple[str, ...]:
    values: List[str]
    if isinstance(raw_value, str):
        values = [part.strip().lower() for part in raw_value.split(",")]
    elif isinstance(raw_value, (list, tuple)):
        values = [str(part).strip().lower() for part in raw_value]
    else:
        raise ValueError(
            "--augmented-filename-markers must be a comma-separated string or sequence of strings."
        )

    markers = tuple(dict.fromkeys(value for value in values if value != ""))
    if len(markers) == 0:
        raise ValueError("--augmented-filename-markers cannot be empty.")
    return markers


def is_augmented_image_entry(image_entry: Dict[str, Any], markers: Sequence[str]) -> bool:
    explicit_flag = image_entry.get("is_augmented")
    if explicit_flag is not None:
        return bool(explicit_flag)
    file_name = str(image_entry.get("file_name", "")).lower()
    return any(marker in file_name for marker in markers)


def infer_augmentation_train_images_root(images_root: Path) -> Path:
    train_dir = images_root / "train"
    if train_dir.exists():
        return train_dir
    return images_root


def remap_path_under_new_root(path: Path, source_root: Path, target_root: Path) -> Path:
    try:
        rel = path.resolve().relative_to(source_root.resolve())
    except ValueError:
        return path
    return target_root / rel


def _linear_sum_assignment_np(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact Hungarian (Kuhn-Munkres) solver for rectangular cost matrices.
    Returns row/col indices of the optimal 1-to-1 assignment.
    """
    if cost_matrix.ndim != 2:
        raise ValueError(f"Expected 2D cost matrix, got shape {cost_matrix.shape}")

    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    transposed = False
    cost = cost_matrix
    if n_rows > n_cols:
        cost = cost.T
        n_rows, n_cols = cost.shape
        transposed = True

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            row = cost[i0 - 1]

            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1

            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = []
    col_ind = []
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            row_ind.append(p[j] - 1)
            col_ind.append(j - 1)

    row_ind_np = np.asarray(row_ind, dtype=np.int64)
    col_ind_np = np.asarray(col_ind, dtype=np.int64)

    order = np.argsort(row_ind_np)
    row_ind_np = row_ind_np[order]
    col_ind_np = col_ind_np[order]

    if transposed:
        row_ind_np, col_ind_np = col_ind_np, row_ind_np

    return row_ind_np, col_ind_np


def linear_sum_assignment_torch(cost_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cost_np = cost_matrix.detach().to(torch.float64).cpu().numpy()
    cost_np = np.nan_to_num(cost_np, nan=1e6, posinf=1e6, neginf=-1e6)
    row_ind, col_ind = _linear_sum_assignment_np(cost_np)
    return (
        torch.as_tensor(row_ind, dtype=torch.long, device=cost_matrix.device),
        torch.as_tensor(col_ind, dtype=torch.long, device=cost_matrix.device),
    )


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bs = outputs["pred_logits"].shape[0]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]
            if tgt_ids.numel() == 0:
                empty = torch.empty((0,), dtype=torch.long, device=out_prob.device)
                indices.append((empty, empty))
                continue

            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_bbox),
            )
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            q_idx, t_idx = linear_sum_assignment_torch(cost)
            indices.append((q_idx, t_idx))
        return indices


class DetectionCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        eos_coef: float = 0.1,
        include_no_object: bool = True,
    ) -> None:
        super().__init__()
        if not include_no_object:
            raise ValueError("Training criterion expects model include_no_object=True.")
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.include_no_object = include_no_object
        self.no_object_class = num_classes

    @staticmethod
    def _get_src_permutation_idx(indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(indices) == 0 or all(src.numel() == 0 for src, _ in indices):
            return (
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices) if src.numel() > 0]
        )
        src_idx = torch.cat([src for (src, _) in indices if src.numel() > 0])
        return batch_idx, src_idx

    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        src_logits = outputs["pred_logits"]
        bs, num_queries, class_dim = src_logits.shape
        target_classes = torch.full(
            (bs, num_queries),
            self.no_object_class,
            dtype=torch.long,
            device=src_logits.device,
        )

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]

        empty_weight = torch.ones(class_dim, device=src_logits.device)
        if self.no_object_class < class_dim:
            empty_weight[self.no_object_class] = self.eos_coef

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=empty_weight)
        return {"loss_ce": loss_ce}

    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][j] for t, (_, j) in zip(targets, indices) if j.numel() > 0],
            dim=0,
        )

        num_boxes = max(target_boxes.shape[0], 1)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes

        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = generalized_box_iou(src_xyxy, tgt_xyxy)
        loss_giou = (1.0 - torch.diag(giou)).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        main_outputs = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }
        indices = self.matcher(main_outputs, targets)

        losses = {}
        losses.update(self.loss_labels(main_outputs, targets, indices))
        losses.update(self.loss_boxes(main_outputs, targets, indices))

        if "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_main = {
                    "pred_logits": aux_out["pred_logits"],
                    "pred_boxes": aux_out["pred_boxes"],
                }
                aux_indices = self.matcher(aux_main, targets)
                aux_losses = {}
                aux_losses.update(self.loss_labels(aux_main, targets, aux_indices))
                aux_losses.update(self.loss_boxes(aux_main, targets, aux_indices))
                for k, v in aux_losses.items():
                    losses[f"{k}_{i}"] = v

        return losses


def loss_weight(name: str, weight_dict: Dict[str, float]) -> Optional[float]:
    for base_name, weight in weight_dict.items():
        if name == base_name or name.startswith(f"{base_name}_"):
            return weight
    return None


def reduce_loss_dict(loss_dict: Dict[str, torch.Tensor], weight_dict: Dict[str, float]) -> torch.Tensor:
    total = None
    for name, value in loss_dict.items():
        w = loss_weight(name, weight_dict)
        if w is None:
            continue
        contrib = value * w
        total = contrib if total is None else total + contrib
    if total is None:
        raise RuntimeError("No weighted losses found. Check weight dict and criterion outputs.")
    return total


def targets_to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device) -> List[Dict[str, torch.Tensor]]:
    out: List[Dict[str, torch.Tensor]] = []
    for target in targets:
        out.append({k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()})
    return out


def build_optimizer(
    model: DINOv2DINOModel,
    optimizer_name: str,
    lr: float,
    backbone_lr_mult: float,
    weight_decay: float,
    soap_args: Dict[str, Any],
    muon_args: Dict[str, Any],
) -> OptimizerLike:
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    other_params = [
        p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad
    ]

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * backbone_lr_mult})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "kl_shampoo":
        return build_kl_shampoo_optimizer(
            params=param_groups,
            lr=lr,
            betas=soap_args["betas"],
            shampoo_beta=soap_args["shampoo_beta"],
            eps=soap_args["eps"],
            weight_decay=weight_decay,
            weight_decay_method=soap_args["weight_decay_method"],
            nesterov=soap_args["nesterov"],
            precondition_frequency=soap_args["precondition_frequency"],
            adam_warmup_steps=soap_args["adam_warmup_steps"],
            correct_bias=soap_args["correct_bias"],
            fp32_matmul_prec=soap_args["fp32_matmul_prec"],
            use_eigh=soap_args["use_eigh"],
            qr_fp32_matmul_prec=soap_args["qr_fp32_matmul_prec"],
            use_adaptive_criteria=soap_args["use_adaptive_criteria"],
            adaptive_update_tolerance=soap_args["adaptive_update_tolerance"],
            power_iter_steps=soap_args["power_iter_steps"],
            max_update_rms=soap_args["max_update_rms"],
            use_kl_shampoo=soap_args["use_kl_shampoo"],
            correct_shampoo_beta_bias=soap_args["correct_shampoo_beta_bias"],
            matrix_only=True,
        )
    if optimizer_name == "muon":
        return build_muon_optimizer(
            params=param_groups,
            lr=lr,
            weight_decay=weight_decay,
            muon_lr_mult=muon_args["lr_mult"],
            muon_momentum=muon_args["momentum"],
            adam_betas=muon_args["adam_betas"],
            adam_eps=muon_args["adam_eps"],
        )
    raise ValueError(f"Unknown optimizer '{optimizer_name}'.")


def build_schedulers(optimizer: OptimizerLike, t_max: int) -> List[torch.optim.lr_scheduler.LRScheduler]:
    if isinstance(optimizer, MultiOptimizer):
        return [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max) for opt in optimizer.optimizers]
    return [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)]


def set_backbone_trainable(model: DINOv2DINOModel, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def run_epoch(
    model: DINOv2DINOModel,
    criterion: DetectionCriterion,
    data_loader: DataLoader,
    optimizer: Optional[OptimizerLike],
    device: torch.device,
    weight_dict: Dict[str, float],
    print_freq: int,
    grad_clip_norm: float,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_steps = 0

    for step, (images, targets) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets_to_device(targets, device)

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = reduce_loss_dict(loss_dict, weight_dict)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1

        if step % print_freq == 0 or step == 1:
            mode = "train" if is_train else "val"
            print(f"[{mode}] step={step}/{len(data_loader)} loss={loss.item():.4f}")

    return total_loss / max(total_steps, 1)


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(x2 - x1, a_min=0.0, a_max=None)
    inter_h = np.clip(y2 - y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area_box = np.clip(box[2] - box[0], a_min=0.0, a_max=None) * np.clip(box[3] - box[1], a_min=0.0, a_max=None)
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], a_min=0.0, a_max=None) * np.clip(
        boxes[:, 3] - boxes[:, 1], a_min=0.0, a_max=None
    )
    union = area_box + area_boxes - inter
    return inter / np.clip(union, a_min=1e-12, a_max=None)


def cxcywh_normalized_to_xyxy_absolute(boxes: torch.Tensor, height: float, width: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    scale = boxes.new_tensor([width, height, width, height])
    xyxy = box_cxcywh_to_xyxy(boxes) * scale
    xyxy[..., 0::2] = xyxy[..., 0::2].clamp(0.0, width)
    xyxy[..., 1::2] = xyxy[..., 1::2].clamp(0.0, height)
    return xyxy


def average_precision_101(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if num_gt <= 0:
        return float("nan")
    if tp.size == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / float(num_gt)
    precisions = tp_cum / np.clip(tp_cum + fp_cum, a_min=1e-12, a_max=None)

    recall_points = np.linspace(0.0, 1.0, 101)
    precision_interp = np.zeros_like(recall_points, dtype=np.float32)
    for i, recall_thr in enumerate(recall_points):
        valid = np.where(recalls >= recall_thr)[0]
        precision_interp[i] = np.max(precisions[valid]) if valid.size > 0 else 0.0
    return float(np.mean(precision_interp))


def compute_coco_style_map(
    predictions_by_class: Dict[int, List[Tuple[int, float, np.ndarray]]],
    targets_by_class: Dict[int, Dict[int, List[np.ndarray]]],
    category_ids: Sequence[int],
    iou_thresholds: Sequence[float],
) -> Dict[str, Any]:
    ap_per_iou: Dict[float, float] = {}
    total_gt_boxes = sum(
        len(boxes) for category_targets in targets_by_class.values() for boxes in category_targets.values()
    )
    categories_with_gt = [
        category_id
        for category_id in category_ids
        if sum(len(v) for v in targets_by_class.get(category_id, {}).values()) > 0
    ]

    for iou_thr in iou_thresholds:
        class_aps: List[float] = []
        for category_id in category_ids:
            gt_by_image = targets_by_class.get(category_id, {})
            num_gt = sum(len(v) for v in gt_by_image.values())
            if num_gt == 0:
                continue

            preds = sorted(predictions_by_class.get(category_id, []), key=lambda item: item[1], reverse=True)
            matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_by_image.items()}
            tp = np.zeros((len(preds),), dtype=np.float32)
            fp = np.zeros((len(preds),), dtype=np.float32)

            for pred_idx, (image_id, _score, pred_box) in enumerate(preds):
                gt_boxes_list = gt_by_image.get(image_id, [])
                if len(gt_boxes_list) == 0:
                    fp[pred_idx] = 1.0
                    continue

                gt_boxes = np.asarray(gt_boxes_list, dtype=np.float32)
                ious = box_iou_xyxy(pred_box, gt_boxes)
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])

                if best_iou >= iou_thr and not matched[image_id][best_idx]:
                    tp[pred_idx] = 1.0
                    matched[image_id][best_idx] = True
                else:
                    fp[pred_idx] = 1.0

            class_aps.append(average_precision_101(tp, fp, num_gt))

        if class_aps:
            ap_per_iou[iou_thr] = float(np.mean(class_aps))
        else:
            ap_per_iou[iou_thr] = float("nan")

    valid_maps = [value for value in ap_per_iou.values() if not math.isnan(value)]
    map_50_95 = float(np.mean(valid_maps)) if valid_maps else float("nan")
    return {
        "map_50_95": map_50_95,
        "ap50": ap_per_iou.get(0.50, float("nan")),
        "ap75": ap_per_iou.get(0.75, float("nan")),
        "ap_per_iou": ap_per_iou,
        "evaluated_categories": len(categories_with_gt),
        "num_gt_boxes": total_gt_boxes,
    }


def evaluate_map_on_loader(
    model: DINOv2DINOModel,
    data_loader: DataLoader,
    device: torch.device,
    ann_file: Path,
    num_classes: int,
    include_no_object: bool,
    score_threshold: float,
    topk: int,
    split_name: str = "validation",
) -> Dict[str, Any]:
    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for COCO-style evaluation. Install it with "
            "`pip install pycocotools` to match starter_pack.ipynb evaluation."
        ) from exc

    model.eval()
    coco_gt = COCO(str(ann_file))
    valid_category_ids = sorted(int(cat["id"]) for cat in coco_gt.dataset.get("categories", []))
    valid_category_id_set = set(valid_category_ids)
    single_category_id = valid_category_ids[0] if len(valid_category_ids) == 1 else None
    detections: List[Dict[str, Any]] = []
    eval_image_ids: List[int] = []

    with torch.no_grad():
        for step, (images, targets) in enumerate(data_loader, start=1):
            images = images.to(device, non_blocking=True)
            outputs = model(images)

            logits = outputs["pred_logits"].detach().cpu()
            boxes = outputs["pred_boxes"].detach().cpu()

            probs = logits.softmax(dim=-1)
            if include_no_object:
                probs = probs[..., :num_classes]

            scores, labels = probs.max(dim=-1)
            if topk > 0 and scores.shape[1] > topk:
                top_scores, top_indices = torch.topk(scores, k=topk, dim=1)
                gathered_labels = torch.gather(labels, 1, top_indices)
                gather_idx = top_indices.unsqueeze(-1).expand(-1, -1, 4)
                gathered_boxes = torch.gather(boxes, 1, gather_idx)
                scores = top_scores
                labels = gathered_labels
                boxes = gathered_boxes

            for batch_idx, target in enumerate(targets):
                image_id = int(target["image_id"].item())
                eval_image_ids.append(image_id)
                resized_h = float(target["size"][0].item())
                resized_w = float(target["size"][1].item())
                orig_h = float(target["orig_size"][0].item())
                orig_w = float(target["orig_size"][1].item())
                scale_x = orig_w / max(resized_w, 1e-6)
                scale_y = orig_h / max(resized_h, 1e-6)

                pred_scores = scores[batch_idx]
                pred_labels = labels[batch_idx]
                pred_boxes_xyxy = cxcywh_normalized_to_xyxy_absolute(boxes[batch_idx], resized_h, resized_w)
                pred_boxes_xyxy[..., 0::2] = (pred_boxes_xyxy[..., 0::2] * scale_x).clamp(0.0, orig_w)
                pred_boxes_xyxy[..., 1::2] = (pred_boxes_xyxy[..., 1::2] * scale_y).clamp(0.0, orig_h)
                pred_boxes_xyxy_np = pred_boxes_xyxy.numpy()
                keep = pred_scores >= score_threshold
                keep_indices = torch.nonzero(keep, as_tuple=False).flatten().tolist()
                for pred_idx in keep_indices:
                    pred_label = int(pred_labels[pred_idx].item())
                    if pred_label in valid_category_id_set:
                        category_id = pred_label
                    elif single_category_id is not None:
                        # Mirror starter pack behavior for single-class RFI detection.
                        category_id = single_category_id
                    else:
                        continue

                    x1, y1, x2, y2 = pred_boxes_xyxy_np[pred_idx].tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    detections.append(
                        {
                            "image_id": image_id,
                            "category_id": int(category_id),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(pred_scores[pred_idx].item()),
                        }
                    )

            if step % 50 == 0 or step == 1:
                print(f"[eval] step={step}/{len(data_loader)}")

    unique_eval_image_ids = sorted(set(eval_image_ids))
    if len(detections) == 0:
        print(
            f"[{split_name}] No detections above score threshold {score_threshold:.4f}. "
            "Reporting zero mAP."
        )
        print("mAP metric:", 0.0)
        return {
            "map_50_95": 0.0,
            "ap50": 0.0,
            "ap75": 0.0,
            "stats": [0.0] * 12,
            "num_detections": 0,
            "num_eval_images": len(unique_eval_image_ids),
        }

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = unique_eval_image_ids
    if valid_category_ids:
        evaluator.params.catIds = valid_category_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    print("mAP metric:", float(evaluator.stats[0]))

    stats = [float(v) for v in evaluator.stats]
    return {
        "map_50_95": stats[0],
        "ap50": stats[1],
        "ap75": stats[2],
        "stats": stats,
        "num_detections": len(detections),
        "num_eval_images": len(unique_eval_image_ids),
    }


def infer_num_classes(annotation_file: Path) -> int:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    cat_ids = sorted(int(c["id"]) for c in data["categories"])
    if not cat_ids:
        raise ValueError("No categories found in annotation file.")
    expected = list(range(min(cat_ids), max(cat_ids) + 1))
    if cat_ids != expected:
        raise ValueError(
            f"Category IDs in {annotation_file} are not contiguous: {cat_ids}"
        )
    return max(cat_ids) + 1


def save_checkpoint(
    output_dir: Path,
    phase: str,
    epoch_in_phase: int,
    global_epoch: int,
    model: DINOv2DINOModel,
    optimizer: OptimizerLike,
    schedulers: Sequence[torch.optim.lr_scheduler.LRScheduler],
    train_loss: float,
    val_loss: float,
    args_dict: Dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{phase}_epoch_{global_epoch:03d}.pth"
    payload = {
        "phase": phase,
        "epoch_in_phase": epoch_in_phase,
        "global_epoch": global_epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedulers": [scheduler.state_dict() for scheduler in schedulers],
        "train_loss": train_loss,
        "val_loss": val_loss,
        "args": args_dict,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def serialize_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _coerce_wandb_value(key: str, current_value: Any, new_value: Any) -> Any:
    if key in PATH_CONFIG_KEYS and new_value is not None:
        return Path(new_value)
    if isinstance(current_value, Path) and new_value is not None:
        return Path(new_value)
    if key == "soap_betas" and isinstance(new_value, (list, tuple)):
        return tuple(float(v) for v in new_value)
    if key == "muon_adam_betas" and isinstance(new_value, (list, tuple)):
        return tuple(float(v) for v in new_value)
    if key == "spectral_fuse_level_indices" and isinstance(new_value, (list, tuple)):
        return tuple(int(v) for v in new_value)
    if isinstance(current_value, tuple) and isinstance(new_value, (list, tuple)):
        return tuple(new_value)
    return new_value


def apply_wandb_sweep_config(args: argparse.Namespace, wandb_config: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, value in dict(wandb_config).items():
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        coerced = _coerce_wandb_value(key, current, value)
        setattr(args, key, coerced)
        overrides[key] = coerced
    return overrides


def maybe_init_wandb(args: argparse.Namespace) -> tuple[Any, Dict[str, Any]]:
    if not args.wandb_enable or args.wandb_mode == "disabled":
        return None, {}

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install it with `pip install wandb` "
            "or add wandb to your project dependencies."
        ) from exc

    init_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "group": args.wandb_group,
        "job_type": args.wandb_job_type,
        "tags": args.wandb_tags,
        "mode": args.wandb_mode,
        "config": serialize_args(args),
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    run = wandb.init(**init_kwargs)
    sweep_overrides: Dict[str, Any] = {}
    if args.wandb_use_sweep_config:
        sweep_overrides = apply_wandb_sweep_config(args, dict(run.config))
        if sweep_overrides:
            keys = ", ".join(sorted(sweep_overrides.keys()))
            print(f"Applied W&B sweep overrides for keys: {keys}")
    return run, sweep_overrides


def collect_lr_metrics(optimizer: OptimizerLike) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if isinstance(optimizer, MultiOptimizer):
        for opt_idx, opt in enumerate(optimizer.optimizers):
            for group_idx, group in enumerate(opt.param_groups):
                metrics[f"lr/opt{opt_idx}_group{group_idx}"] = float(group["lr"])
    else:
        for group_idx, group in enumerate(optimizer.param_groups):
            metrics[f"lr/group{group_idx}"] = float(group["lr"])
    return metrics


def parse_spectral_fuse_level_indices(raw_value: Any) -> Tuple[int, ...]:
    values: List[Any]
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return tuple()
        values = [part.strip() for part in stripped.split(",") if part.strip() != ""]
    elif isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        raise ValueError(
            "--spectral-fuse-level-indices must be a comma-separated string or sequence of ints."
        )

    parsed: List[int] = []
    for value in values:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid fused level index '{value}' in --spectral-fuse-level-indices. "
                "Expected integers like '1,2,3'."
            ) from exc
    return tuple(parsed)


def validate_spectral_args(args: argparse.Namespace) -> Tuple[int, ...]:
    if args.spectral_branch_channels <= 0:
        raise ValueError(
            f"--spectral-branch-channels must be a positive integer, got {args.spectral_branch_channels}."
        )
    if args.spectral_patch_size <= 0:
        raise ValueError(f"--spectral-patch-size must be a positive integer, got {args.spectral_patch_size}.")
    if args.spectral_dct_keep_size <= 0:
        raise ValueError(
            f"--spectral-dct-keep-size must be a positive integer, got {args.spectral_dct_keep_size}."
        )
    if args.spectral_dct_keep_size > args.spectral_patch_size:
        raise ValueError(
            f"--spectral-dct-keep-size ({args.spectral_dct_keep_size}) cannot exceed "
            f"--spectral-patch-size ({args.spectral_patch_size})."
        )
    if args.spectral_drop_dc and args.spectral_dct_keep_size == 1:
        raise ValueError(
            "--spectral-drop-dc with --spectral-dct-keep-size=1 removes all coefficients. "
            "Increase --spectral-dct-keep-size or disable --spectral-drop-dc."
        )

    fuse_indices = parse_spectral_fuse_level_indices(args.spectral_fuse_level_indices)
    if len(set(fuse_indices)) != len(fuse_indices):
        raise ValueError(
            f"--spectral-fuse-level-indices contains duplicates: {fuse_indices}. "
            "Provide each level at most once."
        )
    if not args.no_spectral_prior and len(fuse_indices) == 0:
        raise ValueError(
            "--spectral-fuse-level-indices cannot be empty when spectral prior is enabled. "
            "Use --no-spectral-prior for ablations."
        )
    for level_idx in fuse_indices:
        if level_idx < 0 or level_idx >= args.num_feature_levels:
            raise ValueError(
                f"--spectral-fuse-level-indices contains out-of-range level {level_idx}. "
                f"Valid range is [0, {args.num_feature_levels - 1}] for --num-feature-levels={args.num_feature_levels}."
            )
    return fuse_indices


def main() -> None:
    args = parse_args()
    wandb_run, _ = maybe_init_wandb(args)
    set_seed(args.seed)
    augmented_filename_markers = parse_augmented_filename_markers(args.augmented_filename_markers)
    args.augmented_filename_markers = augmented_filename_markers

    default_train_ann_candidates = [
        args.data_root / "annotations" / "instances_train.json",
        args.data_root / "annotations" / "train.json",
    ]
    default_val_ann_candidates = [
        args.data_root / "annotations" / "instances_val.json",
        args.data_root / "annotations" / "val.json",
    ]
    default_test_ann_candidates = [
        args.data_root / "annotations" / "instances_test.json",
        args.data_root / "annotations" / "test.json",
    ]

    if args.train_ann is None:
        for candidate in default_train_ann_candidates:
            if candidate.exists():
                args.train_ann = candidate
                break
        if args.train_ann is None:
            raise FileNotFoundError(
                "Could not infer training annotation path. Looked for: "
                + ", ".join(str(path) for path in default_train_ann_candidates)
            )
    if args.val_ann is None:
        for candidate in default_val_ann_candidates:
            if candidate.exists():
                args.val_ann = candidate
                break
        if args.val_ann is None:
            if args.no_auto_train_val_split:
                args.val_ann = args.train_ann
                print(
                    "No validation annotation file found. Falling back to training annotations for validation "
                    "(smoke-test mode)."
                )
            else:
                split_train_ann, split_val_ann = create_train_val_split_annotations(
                    source_ann=args.train_ann,
                    val_ratio=args.train_val_split_ratio,
                    split_seed=args.train_val_split_seed,
                    val_originals_only=args.val_originals_only,
                    augmented_filename_markers=augmented_filename_markers,
                )
                args.train_ann = split_train_ann
                args.val_ann = split_val_ann
                print(
                    "No validation annotation file found. Generated deterministic train/val split: "
                    f"train={args.train_ann.name}, val={args.val_ann.name} "
                    f"(ratio={1.0 - args.train_val_split_ratio:.2f}/{args.train_val_split_ratio:.2f}, "
                    f"seed={args.train_val_split_seed})."
                )
    if args.images_root is None:
        args.images_root = args.data_root / "images"
    if args.test_ann is None:
        for candidate in default_test_ann_candidates:
            if candidate.exists():
                args.test_ann = candidate
                break
    if args.test_ann is None:
        args.test_ann = args.val_ann
        print(
            "No test annotation file found. Final mAP will be reported on validation annotations."
        )
    if args.test_images_root is None:
        if "test" in args.test_ann.stem.lower():
            candidate_test_root = args.data_root / "images" / "test"
            if candidate_test_root.exists():
                args.test_images_root = candidate_test_root
        if args.test_images_root is None:
            args.test_images_root = args.images_root
    source_data_root_for_remap = args.data_root
    if args.augment_train_after_split:
        augmentation_output_root = (
            args.augmentation_output_root
            if args.augmentation_output_root is not None
            else args.output_dir / "runtime_augmented_data"
        )
        augmentation_train_images_root = infer_augmentation_train_images_root(args.images_root)
        augmentation_output_ann = augmentation_output_root / "annotations" / args.train_ann.name

        augment_summary = augment_train_dataset(
            source_data_root=source_data_root_for_remap,
            output_data_root=augmentation_output_root,
            train_ann=args.train_ann,
            images_root=augmentation_train_images_root,
            output_ann=augmentation_output_ann,
            output_images_dir=augmentation_output_root / "images" / "train",
            crop_scale=args.augmentation_crop_scale,
            min_visible_frac=args.augmentation_min_visible_frac,
            min_box_area=args.augmentation_min_box_area,
            seed=args.augmentation_seed,
            include_original=args.augmentation_include_original,
            overwrite=args.augmentation_overwrite,
            strict=args.augmentation_strict,
        )

        args.augmentation_output_root = augmentation_output_root
        args.data_root = augmentation_output_root
        args.train_ann = Path(augment_summary["output_ann"])
        args.val_ann = remap_path_under_new_root(args.val_ann, source_data_root_for_remap, augmentation_output_root)
        args.test_ann = remap_path_under_new_root(args.test_ann, source_data_root_for_remap, augmentation_output_root)
        args.images_root = remap_path_under_new_root(args.images_root, source_data_root_for_remap, augmentation_output_root)
        args.test_images_root = remap_path_under_new_root(
            args.test_images_root,
            source_data_root_for_remap,
            augmentation_output_root,
        )
        print(
            "Runtime augmentation enabled: "
            f"train_ann={args.train_ann} val_ann={args.val_ann} "
            f"images_root={args.images_root} data_root={args.data_root}"
        )
    if args.image_size <= 0:
        raise ValueError("--image-size must be a positive integer for DINOv2 fine-tuning.")
    if args.image_size % DINOV2_PATCH_SIZE != 0:
        raise ValueError(
            f"--image-size must be divisible by patch size {DINOV2_PATCH_SIZE}, got {args.image_size}."
        )
    if not (0.0 < args.train_val_split_ratio < 1.0):
        raise ValueError(
            f"--train-val-split-ratio must be in (0, 1), got {args.train_val_split_ratio}."
        )
    if not (0.0 <= args.eval_score_threshold <= 1.0):
        raise ValueError(
            f"--eval-score-threshold must be in [0, 1], got {args.eval_score_threshold}."
        )
    if args.eval_topk <= 0:
        raise ValueError(f"--eval-topk must be > 0, got {args.eval_topk}.")
    fused_level_indices = validate_spectral_args(args)
    args.spectral_fuse_level_indices = fused_level_indices

    if args.num_classes is None:
        args.num_classes = infer_num_classes(args.train_ann)
        print(f"Inferred num_classes={args.num_classes} from {args.train_ann}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = ClearSARCocoDataset(
        images_root=args.images_root,
        ann_file=args.train_ann,
        image_size=args.image_size,
        originals_only=False,
        augmented_filename_markers=augmented_filename_markers,
    )
    val_dataset = ClearSARCocoDataset(
        images_root=args.images_root,
        ann_file=args.val_ann,
        image_size=args.image_size,
        originals_only=args.val_originals_only,
        augmented_filename_markers=augmented_filename_markers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if args.backbone_checkpoint is None:
        print(f"Loading pretrained DINOv2 backbone '{args.backbone_name}' from timm weights.")
    else:
        print(f"Loading DINOv2 backbone '{args.backbone_name}' from local checkpoint: {args.backbone_checkpoint}")

    model = DINOv2DINOModel(
        backbone_name=args.backbone_name,
        backbone_checkpoint_path=args.backbone_checkpoint,
        backbone_pretrained=args.backbone_checkpoint is None,
        neck_out_channels=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        with_box_refine=not args.no_box_refine,
        two_stage=not args.no_two_stage,
        include_no_object=args.include_no_object,
        freeze_backbone=False,
        use_spectral_prior=not args.no_spectral_prior,
        spectral_fuse_level_indices=fused_level_indices,
        spectral_branch_channels=args.spectral_branch_channels,
        spectral_patch_size=args.spectral_patch_size,
        spectral_dct_keep_size=args.spectral_dct_keep_size,
        spectral_drop_dc=args.spectral_drop_dc,
    ).to(device)
    if wandb_run is not None:
        wandb_run.config.update(serialize_args(args), allow_val_change=True)
        if args.wandb_watch:
            wandb_run.watch(model, log="all", log_freq=args.wandb_log_freq)

    matcher = HungarianMatcher(
        cost_class=args.cls_loss_coef,
        cost_bbox=args.bbox_loss_coef,
        cost_giou=args.giou_loss_coef,
    )
    criterion = DetectionCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        eos_coef=args.eos_coef,
        include_no_object=args.include_no_object,
    ).to(device)

    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }

    phases = [
        ("phase_a", args.phase_a_epochs, True, args.backbone_lr_mult),
        ("phase_b", args.phase_b_epochs, False, args.backbone_lr_mult),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args_dict = serialize_args(args)
    (args.output_dir / "config.json").write_text(json.dumps(args_dict, indent=2), encoding="utf-8")

    best_path = args.output_dir / "best.pth"
    best_val = math.inf
    if best_path.exists():
        try:
            best_payload = torch.load(best_path, map_location="cpu")
            if isinstance(best_payload, dict) and "best_val_loss" in best_payload:
                best_val = float(best_payload["best_val_loss"])
                print(f"Loaded existing best validation loss from {best_path}: {best_val:.4f}")
        except Exception as exc:
            print(f"Warning: failed to read existing best checkpoint at {best_path}: {exc}")

    global_epoch = 0
    resume_phase_idx: Optional[int] = None
    resume_epoch_in_phase = 0
    resume_optimizer_state: Optional[Dict[str, Any]] = None
    resume_scheduler_states: Optional[List[Dict[str, Any]]] = None
    phase_name_to_idx = {phase_name: idx for idx, (phase_name, _, _, _) in enumerate(phases)}

    if args.resume_checkpoint is not None:
        resume_payload = torch.load(args.resume_checkpoint, map_location=device)
        if not isinstance(resume_payload, dict):
            raise ValueError(
                f"Checkpoint payload must be a dictionary, got {type(resume_payload).__name__} "
                f"from {args.resume_checkpoint}."
            )
        if "model" not in resume_payload:
            raise KeyError(f"Checkpoint {args.resume_checkpoint} does not contain a 'model' state dict.")
        if "optimizer" not in resume_payload:
            raise KeyError(
                f"Checkpoint {args.resume_checkpoint} does not contain an 'optimizer' state dict. "
                "Use a per-epoch training checkpoint (e.g., phase_a_epoch_XXX.pth), not best.pth."
            )
        phase_name = resume_payload.get("phase")
        if not isinstance(phase_name, str) or phase_name not in phase_name_to_idx:
            valid_phase_names = tuple(phase_name_to_idx.keys())
            raise ValueError(
                f"Checkpoint {args.resume_checkpoint} has invalid phase '{phase_name}'. "
                f"Expected one of {valid_phase_names}."
            )

        model.load_state_dict(resume_payload["model"], strict=True)
        global_epoch = int(resume_payload.get("global_epoch", 0))
        resume_phase_idx = phase_name_to_idx[phase_name]
        resume_epoch_in_phase = int(resume_payload.get("epoch_in_phase", 0))
        if resume_epoch_in_phase < 0:
            raise ValueError(f"Invalid epoch_in_phase={resume_epoch_in_phase} in {args.resume_checkpoint}.")
        resume_optimizer_state = resume_payload["optimizer"]
        scheduler_states_raw = resume_payload.get("schedulers")
        if isinstance(scheduler_states_raw, list):
            resume_scheduler_states = scheduler_states_raw

        checkpoint_val_loss = resume_payload.get("val_loss")
        if checkpoint_val_loss is not None:
            best_val = min(best_val, float(checkpoint_val_loss))

        print(
            "Resuming from checkpoint: "
            f"{args.resume_checkpoint} | phase={phase_name} | "
            f"epoch_in_phase={resume_epoch_in_phase} | global_epoch={global_epoch}"
        )

    correct_shampoo_beta_bias: Optional[bool]
    if args.soap_correct_shampoo_beta_bias == "none":
        correct_shampoo_beta_bias = None
    else:
        correct_shampoo_beta_bias = args.soap_correct_shampoo_beta_bias == "true"

    soap_args: Dict[str, Any] = {
        "betas": tuple(args.soap_betas),
        "shampoo_beta": args.soap_shampoo_beta,
        "eps": args.soap_eps,
        "weight_decay_method": args.soap_weight_decay_method,
        "nesterov": args.soap_nesterov,
        "precondition_frequency": args.soap_precondition_frequency,
        "adam_warmup_steps": args.soap_adam_warmup_steps,
        "correct_bias": args.soap_correct_bias,
        "fp32_matmul_prec": args.soap_fp32_matmul_prec,
        "use_eigh": args.soap_use_eigh,
        "qr_fp32_matmul_prec": args.soap_qr_fp32_matmul_prec,
        "use_adaptive_criteria": args.soap_use_adaptive_criteria,
        "adaptive_update_tolerance": args.soap_adaptive_update_tolerance,
        "power_iter_steps": args.soap_power_iter_steps,
        "max_update_rms": args.soap_max_update_rms,
        "use_kl_shampoo": not args.soap_no_kl_shampoo,
        "correct_shampoo_beta_bias": correct_shampoo_beta_bias,
    }
    muon_args: Dict[str, Any] = {
        "lr_mult": args.muon_lr_mult,
        "momentum": args.muon_momentum,
        "adam_betas": tuple(args.muon_adam_betas),
        "adam_eps": args.muon_adam_eps,
    }

    for phase_idx, (phase_name, phase_epochs, freeze_backbone, backbone_lr_mult) in enumerate(phases):
        if phase_epochs <= 0:
            continue

        if resume_phase_idx is not None and phase_idx < resume_phase_idx:
            print(f"Skipping {phase_name}; already completed before resume checkpoint.")
            continue

        start_epoch = 1
        should_restore_phase_state = False
        if resume_phase_idx is not None and phase_idx == resume_phase_idx:
            start_epoch = resume_epoch_in_phase + 1
            if start_epoch > phase_epochs:
                print(
                    f"Skipping {phase_name}; checkpoint already completed this phase "
                    f"({resume_epoch_in_phase}/{phase_epochs} epochs)."
                )
                resume_phase_idx = None
                resume_optimizer_state = None
                resume_scheduler_states = None
                continue
            should_restore_phase_state = True

        set_backbone_trainable(model, trainable=not freeze_backbone)
        optimizer = build_optimizer(
            model=model,
            optimizer_name=args.optimizer,
            lr=args.lr,
            backbone_lr_mult=backbone_lr_mult,
            weight_decay=args.weight_decay,
            soap_args=soap_args,
            muon_args=muon_args,
        )
        schedulers = build_schedulers(optimizer, t_max=phase_epochs)
        if should_restore_phase_state:
            if resume_optimizer_state is None:
                raise RuntimeError("Missing optimizer state while attempting to resume training.")
            optimizer.load_state_dict(resume_optimizer_state)
            if resume_scheduler_states is not None and len(resume_scheduler_states) == len(schedulers):
                for scheduler, scheduler_state in zip(schedulers, resume_scheduler_states):
                    scheduler.load_state_dict(scheduler_state)
            elif resume_epoch_in_phase > 0:
                # Backward compatibility for checkpoints created before scheduler states were saved.
                for _ in range(resume_epoch_in_phase):
                    for scheduler in schedulers:
                        scheduler.step()
            print(
                f"Restored optimizer/scheduler state for {phase_name}; resuming at epoch {start_epoch}/{phase_epochs}."
            )
            resume_phase_idx = None
            resume_optimizer_state = None
            resume_scheduler_states = None

        print(
            f"\n=== {phase_name} | epochs={phase_epochs} | "
            f"backbone_trainable={not freeze_backbone} | "
            f"backbone_lr={args.lr * backbone_lr_mult:.2e} | head_lr={args.lr:.2e} ==="
        )

        for epoch in range(start_epoch, phase_epochs + 1):
            global_epoch += 1
            print(f"\n[{phase_name}] epoch {epoch}/{phase_epochs} (global {global_epoch})")
            train_loss = run_epoch(
                model=model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                weight_dict=weight_dict,
                print_freq=args.print_freq,
                grad_clip_norm=args.grad_clip_norm,
            )
            with torch.no_grad():
                val_loss = run_epoch(
                    model=model,
                    criterion=criterion,
                    data_loader=val_loader,
                    optimizer=None,
                    device=device,
                    weight_dict=weight_dict,
                    print_freq=args.print_freq,
                    grad_clip_norm=args.grad_clip_norm,
                )
            for scheduler in schedulers:
                scheduler.step()

            ckpt_path = save_checkpoint(
                output_dir=args.output_dir,
                phase=phase_name,
                epoch_in_phase=epoch,
                global_epoch=global_epoch,
                model=model,
                optimizer=optimizer,
                schedulers=schedulers,
                train_loss=train_loss,
                val_loss=val_loss,
                args_dict=args_dict,
            )
            print(f"[{phase_name}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} | saved: {ckpt_path}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "phase": phase_name,
                        "global_epoch": global_epoch,
                        "model": model.state_dict(),
                        "best_val_loss": best_val,
                        "args": args_dict,
                    },
                    best_path,
                )
                print(f"New best checkpoint: {best_path} (val_loss={best_val:.4f})")
                if wandb_run is not None:
                    wandb_run.summary["best_val_loss"] = float(best_val)
                    wandb_run.summary["best_checkpoint"] = str(best_path)

            if wandb_run is not None:
                metrics = {
                    "epoch": global_epoch,
                    "phase/name": phase_name,
                    "phase/epoch": epoch,
                    "train/loss": float(train_loss),
                    "val/loss": float(val_loss),
                    "val/best_loss": float(best_val),
                }
                metrics.update(collect_lr_metrics(optimizer))
                wandb_run.log(metrics, step=global_epoch)

    best_path = args.output_dir / "best.pth"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model"], strict=True)
        print(f"Loaded best checkpoint for final mAP evaluation: {best_path}")
    else:
        print("Best checkpoint not found. Evaluating final in-memory model state.")

    eval_dataset = ClearSARCocoDataset(
        images_root=args.test_images_root,
        ann_file=args.test_ann,
        image_size=args.image_size,
        originals_only=args.val_originals_only and "test" not in args.test_ann.stem.lower(),
        augmented_filename_markers=augmented_filename_markers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    eval_split = "test" if "test" in args.test_ann.stem.lower() else "validation"
    print(f"Running COCOeval on {eval_split} split: {args.test_ann}")
    eval_metrics = evaluate_map_on_loader(
        model=model,
        data_loader=eval_loader,
        device=device,
        ann_file=args.test_ann,
        num_classes=args.num_classes,
        include_no_object=args.include_no_object,
        score_threshold=args.eval_score_threshold,
        topk=args.eval_topk,
        split_name=eval_split,
    )

    if wandb_run is not None:
        wandb_run.summary[f"{eval_split}_map_50_95"] = float(eval_metrics["map_50_95"])
        wandb_run.summary[f"{eval_split}_ap50"] = float(eval_metrics["ap50"])
        wandb_run.summary[f"{eval_split}_ap75"] = float(eval_metrics["ap75"])
        wandb_run.summary[f"{eval_split}_num_detections"] = int(eval_metrics["num_detections"])
        wandb_run.summary[f"{eval_split}_num_eval_images"] = int(eval_metrics["num_eval_images"])

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
