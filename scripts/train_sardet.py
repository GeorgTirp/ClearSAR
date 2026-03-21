#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
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

from src.model import SummitDINOModel
from src.optim import build_kl_shampoo_optimizer


PATH_CONFIG_KEYS = {
    "data_root",
    "train_ann",
    "val_ann",
    "images_root",
    "output_dir",
}


def load_config_file(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix.lower() != ".json":
        raise ValueError(f"Only JSON config is supported, got: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object, got: {type(payload).__name__}")

    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        dest = key.replace("-", "_")
        if dest in PATH_CONFIG_KEYS and value is not None:
            normalized[dest] = Path(value)
        else:
            normalized[dest] = value
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SUMMIT + DINO on SARDet with 2-phase schedule.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config file. CLI args override config values.",
    )

    parser.add_argument("--data-root", type=Path, default=Path("src/data/SARDet_COCO"))
    parser.add_argument("--train-ann", type=Path, default=None)
    parser.add_argument("--val-ann", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)

    parser.add_argument("--summit-checkpoint", type=str, default=None)
    parser.add_argument(
        "--backbone-variant",
        type=str,
        default="auto",
        choices=["auto", "base", "large", "huge"],
        help="SUMMIT encoder size. 'auto' infers from checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sardet"))

    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--phase-a-epochs", type=int, default=5)
    parser.add_argument("--phase-b-epochs", type=int, default=45)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "kl_shampoo"])
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

    parser.add_argument("--cls-loss-coef", type=float, default=1.0)
    parser.add_argument("--bbox-loss-coef", type=float, default=5.0)
    parser.add_argument("--giou-loss-coef", type=float, default=2.0)
    parser.add_argument("--eos-coef", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--print-freq", type=int, default=20)

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

    args = parser.parse_args()
    if args.summit_checkpoint is None:
        parser.error("--summit-checkpoint is required (provide via CLI or config).")
    return args


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


class SardetCocoDataset(Dataset):
    def __init__(self, images_root: Path, ann_file: Path, image_size: int) -> None:
        self.images_root = images_root
        self.ann_file = ann_file
        self.image_size = image_size

        data = json.loads(ann_file.read_text(encoding="utf-8"))
        self.images = sorted(data["images"], key=lambda x: x["id"])
        self.categories = sorted(data["categories"], key=lambda x: x["id"])
        self.image_to_anns: Dict[int, List[Dict]] = {}
        for ann in data["annotations"]:
            image_id = int(ann["image_id"])
            self.image_to_anns.setdefault(image_id, []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_info = self.images[idx]
        img_path = self.images_root / image_info["file_name"]
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        if self.image_size > 0:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            out_w = out_h = self.image_size
        else:
            out_w, out_h = orig_w, orig_h

        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

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
    model: SummitDINOModel,
    optimizer_name: str,
    lr: float,
    backbone_lr_mult: float,
    weight_decay: float,
    soap_args: Dict[str, Any],
) -> torch.optim.Optimizer:
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
        )
    raise ValueError(f"Unknown optimizer '{optimizer_name}'.")


def set_backbone_trainable(model: SummitDINOModel, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def run_epoch(
    model: SummitDINOModel,
    criterion: DetectionCriterion,
    data_loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
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
    model: SummitDINOModel,
    optimizer: torch.optim.Optimizer,
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.train_ann is None:
        args.train_ann = args.data_root / "annotations" / "train.json"
    if args.val_ann is None:
        args.val_ann = args.data_root / "annotations" / "val.json"
    if args.images_root is None:
        args.images_root = args.data_root / "images"

    if args.num_classes is None:
        args.num_classes = infer_num_classes(args.train_ann)
        print(f"Inferred num_classes={args.num_classes} from {args.train_ann}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = SardetCocoDataset(
        images_root=args.images_root,
        ann_file=args.train_ann,
        image_size=args.image_size,
    )
    val_dataset = SardetCocoDataset(
        images_root=args.images_root,
        ann_file=args.val_ann,
        image_size=args.image_size,
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

    model = SummitDINOModel(
        summit_checkpoint_path=args.summit_checkpoint,
        backbone_variant=args.backbone_variant,
        backbone_img_size=args.image_size,
        in_chans=3,
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
    ).to(device)

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

    best_val = math.inf
    global_epoch = 0
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

    for phase_name, phase_epochs, freeze_backbone, backbone_lr_mult in phases:
        if phase_epochs <= 0:
            continue

        set_backbone_trainable(model, trainable=not freeze_backbone)
        optimizer = build_optimizer(
            model=model,
            optimizer_name=args.optimizer,
            lr=args.lr,
            backbone_lr_mult=backbone_lr_mult,
            weight_decay=args.weight_decay,
            soap_args=soap_args,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs)

        print(
            f"\n=== {phase_name} | epochs={phase_epochs} | "
            f"backbone_trainable={not freeze_backbone} | "
            f"backbone_lr={args.lr * backbone_lr_mult:.2e} | head_lr={args.lr:.2e} ==="
        )

        for epoch in range(1, phase_epochs + 1):
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
            scheduler.step()

            ckpt_path = save_checkpoint(
                output_dir=args.output_dir,
                phase=phase_name,
                epoch_in_phase=epoch,
                global_epoch=global_epoch,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
                args_dict=args_dict,
            )
            print(f"[{phase_name}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} | saved: {ckpt_path}")

            if val_loss < best_val:
                best_val = val_loss
                best_path = args.output_dir / "best.pth"
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


if __name__ == "__main__":
    main()
