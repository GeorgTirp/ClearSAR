import math
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed


ENCODER_PREFIXES = ("patch_embed.", "cls_token", "pos_embed", "blocks.", "norm.")


def _extract_encoder_state_from_checkpoint(
    checkpoint_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    encoder_state: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint_state.items():
        cleaned_key = key.removeprefix("module.")
        if cleaned_key.startswith(ENCODER_PREFIXES):
            encoder_state[cleaned_key] = value
    return encoder_state


def _load_checkpoint_state_dict(checkpoint_file: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_file}.")
    return state


def _infer_num_heads(embed_dim: int, depth: int) -> int:
    known = {
        (768, 12): 12,
        (1024, 24): 16,
        (1280, 32): 16,
    }
    if (embed_dim, depth) in known:
        return known[(embed_dim, depth)]
    return max(embed_dim // 64, 1)


def infer_summit_backbone_config(checkpoint_path: str) -> Dict[str, int]:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if _is_lfs_pointer(checkpoint_file):
        raise RuntimeError(
            f"{checkpoint_file} is a Git LFS pointer file, not real weights. "
            "Pull LFS weights or reconstruct checkpoint from split files first."
        )

    state = _load_checkpoint_state_dict(checkpoint_file)
    encoder_state = _extract_encoder_state_from_checkpoint(state)
    if "cls_token" not in encoder_state or "patch_embed.proj.weight" not in encoder_state:
        raise ValueError(f"Checkpoint does not contain expected SUMMIT encoder keys: {checkpoint_file}")

    embed_dim = int(encoder_state["cls_token"].shape[-1])
    patch_weight = encoder_state["patch_embed.proj.weight"]
    in_chans = int(patch_weight.shape[1])
    patch_size = int(patch_weight.shape[-1])

    block_indices = []
    for key in encoder_state:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
    depth = (max(block_indices) + 1) if block_indices else 0
    num_heads = _infer_num_heads(embed_dim, depth)

    return {
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "patch_size": patch_size,
        "in_chans": in_chans,
    }


def _infer_grid_size(num_patches: int) -> Tuple[int, int]:
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(
            f"Expected square number of patches for positional embedding, got {num_patches}."
        )
    return side, side


def _interpolate_pos_embed(
    pos_embed: torch.Tensor, dst_grid_size: Tuple[int, int]
) -> torch.Tensor:
    cls_pos = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]

    src_h, src_w = _infer_grid_size(patch_pos.shape[1])
    dst_h, dst_w = dst_grid_size
    if (src_h, src_w) == (dst_h, dst_w):
        return pos_embed

    patch_pos = patch_pos.reshape(1, src_h, src_w, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos, size=(dst_h, dst_w), mode="bicubic", align_corners=False
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, dst_h * dst_w, -1)
    return torch.cat([cls_pos, patch_pos], dim=1)


def _is_lfs_pointer(checkpoint_path: Path) -> bool:
    with checkpoint_path.open("rb") as handle:
        header = handle.read(100)
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


class SUMMITBackbone(nn.Module):
    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    @property
    def out_channels(self) -> int:
        return self.patch_embed.proj.weight.shape[0]

    @property
    def stride(self) -> int:
        return self.patch_embed.patch_size[0]

    @property
    def embed_dim(self) -> int:
        return int(self.cls_token.shape[-1])

    @property
    def depth(self) -> int:
        return len(self.blocks)

    def load_summit_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        if _is_lfs_pointer(checkpoint_file):
            raise RuntimeError(
                f"{checkpoint_file} is a Git LFS pointer file, not real weights. "
                "Pull LFS weights or reconstruct checkpoint from split files first."
            )

        state = _load_checkpoint_state_dict(checkpoint_file)
        encoder_state = _extract_encoder_state_from_checkpoint(state)
        if "pos_embed" in encoder_state and encoder_state["pos_embed"].shape != self.pos_embed.shape:
            dst_grid = self.patch_embed.grid_size
            encoder_state["pos_embed"] = _interpolate_pos_embed(encoder_state["pos_embed"], dst_grid)

        try:
            msg = self.load_state_dict(encoder_state, strict=False)
        except RuntimeError as exc:
            inferred = infer_summit_backbone_config(checkpoint_path)
            raise RuntimeError(
                "SUMMIT checkpoint/backbone architecture mismatch. "
                f"Backbone expects embed_dim={self.embed_dim}, depth={self.depth}; "
                f"checkpoint looks like embed_dim={inferred['embed_dim']}, depth={inferred['depth']}. "
                "Use a matching backbone variant (or backbone_variant='auto')."
            ) from exc
        unexpected = [k for k in msg.unexpected_keys if not k.startswith("decoder")]
        missing = [k for k in msg.missing_keys if not k.startswith("decoder")]
        if unexpected:
            raise RuntimeError(f"Unexpected encoder keys when loading checkpoint: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing encoder keys when loading checkpoint: {missing}")

    def _pos_embed_for_grid(self, grid_h: int, grid_w: int) -> torch.Tensor:
        num_patches = self.pos_embed.shape[1] - 1
        src_h, src_w = _infer_grid_size(num_patches)
        if (src_h, src_w) == (grid_h, grid_w):
            return self.pos_embed
        return _interpolate_pos_embed(self.pos_embed, (grid_h, grid_w))

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        input_h, input_w = x.shape[-2], x.shape[-1]
        patch_h, patch_w = self.patch_embed.patch_size
        grid_h, grid_w = input_h // patch_h, input_w // patch_w
        x = self.patch_embed(x)
        batch_size, num_tokens, _ = x.shape
        if num_tokens != grid_h * grid_w:
            raise RuntimeError(
                f"Patch token count mismatch: got {num_tokens} tokens for expected "
                f"grid {grid_h}x{grid_w}."
            )

        pos_embed = self._pos_embed_for_grid(grid_h, grid_w)
        x = x + pos_embed[:, 1:, :]

        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, grid_h, grid_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, grid_h, grid_w = self.forward_tokens(x)
        patch_tokens = tokens[:, 1:, :]
        features = patch_tokens.transpose(1, 2).reshape(x.shape[0], -1, grid_h, grid_w)
        return features


def summit_vit_large_patch16(img_size: int = 448, in_chans: int = 3) -> SUMMITBackbone:
    return SUMMITBackbone(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def summit_vit_base_patch16(img_size: int = 448, in_chans: int = 3) -> SUMMITBackbone:
    return SUMMITBackbone(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def summit_vit_huge_patch14(img_size: int = 448, in_chans: int = 3) -> SUMMITBackbone:
    return SUMMITBackbone(
        img_size=img_size,
        patch_size=14,
        in_chans=in_chans,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def build_summit_backbone(
    img_size: int = 448,
    in_chans: int = 3,
    backbone_variant: str = "auto",
    checkpoint_path: Optional[str] = None,
) -> SUMMITBackbone:
    variant = backbone_variant.lower()

    if variant == "auto":
        if checkpoint_path is None:
            return summit_vit_base_patch16(img_size=img_size, in_chans=in_chans)
        cfg = infer_summit_backbone_config(checkpoint_path)
        return SUMMITBackbone(
            img_size=img_size,
            patch_size=cfg["patch_size"],
            in_chans=cfg["in_chans"],
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            mlp_ratio=4.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

    if variant == "base":
        return summit_vit_base_patch16(img_size=img_size, in_chans=in_chans)
    if variant == "large":
        return summit_vit_large_patch16(img_size=img_size, in_chans=in_chans)
    if variant == "huge":
        return summit_vit_huge_patch14(img_size=img_size, in_chans=in_chans)
    raise ValueError(
        f"Unknown backbone_variant '{backbone_variant}'. "
        "Expected one of: auto, base, large, huge."
    )
