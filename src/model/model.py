import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv2Backbone, SUMMITBackbone, build_dinov2_backbone, build_summit_backbone
from .detectors import DeformableDINOHead


class SimpleFeaturePyramid(nn.Module):
    """
    ViTDet-style simple feature pyramid that converts one ViT feature map
    into four multi-scale feature levels.
    """

    def __init__(self, in_channels: int, out_channels: int = 256, num_feature_levels: int = 4) -> None:
        super().__init__()
        if num_feature_levels != 4:
            raise ValueError("SimpleFeaturePyramid currently supports exactly 4 feature levels.")

        self.num_feature_levels = num_feature_levels
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(num_feature_levels)]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                for _ in range(num_feature_levels)
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for conv in self.lateral_convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)
        for conv in self.output_convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # From a stride-16 ViT map, produce levels at strides [4, 8, 16, 32].
        resized = [
            F.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False),
            F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False),
            x,
            F.max_pool2d(x, kernel_size=2, stride=2),
        ]
        outputs = []
        for feat, lateral, output in zip(resized, self.lateral_convs, self.output_convs):
            outputs.append(output(lateral(feat)))
        return outputs


def _make_dct_basis(size: int) -> torch.Tensor:
    if size <= 0:
        raise ValueError(f"DCT basis size must be > 0, got {size}.")

    positions = torch.arange(size, dtype=torch.float32)
    frequencies = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos((math.pi / size) * (positions + 0.5) * frequencies)
    basis[0] *= math.sqrt(1.0 / size)
    if size > 1:
        basis[1:] *= math.sqrt(2.0 / size)
    return basis


def _zigzag_indices(block_size: int) -> List[Tuple[int, int]]:
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}.")

    indices: List[Tuple[int, int]] = []
    for diagonal in range(2 * block_size - 1):
        if diagonal % 2 == 0:
            row = min(diagonal, block_size - 1)
            col = diagonal - row
            while row >= 0 and col < block_size:
                indices.append((row, col))
                row -= 1
                col += 1
        else:
            col = min(diagonal, block_size - 1)
            row = diagonal - col
            while col >= 0 and row < block_size:
                indices.append((row, col))
                row += 1
                col -= 1
    return indices


class SpectralPriorBranch(nn.Module):
    def __init__(
        self,
        num_feature_levels: int,
        branch_channels: int = 64,
        patch_size: int = 8,
        dct_keep_size: int = 4,
        drop_dc: bool = False,
    ) -> None:
        super().__init__()
        if num_feature_levels <= 0:
            raise ValueError(f"num_feature_levels must be > 0, got {num_feature_levels}.")
        if branch_channels <= 0:
            raise ValueError(f"branch_channels must be > 0, got {branch_channels}.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}.")
        if dct_keep_size <= 0:
            raise ValueError(f"dct_keep_size must be > 0, got {dct_keep_size}.")
        if dct_keep_size > patch_size:
            raise ValueError(f"dct_keep_size ({dct_keep_size}) cannot exceed patch_size ({patch_size}).")

        zigzag_block = _zigzag_indices(dct_keep_size)
        coeff_indices = [row * patch_size + col for row, col in zigzag_block]
        if drop_dc:
            coeff_indices = coeff_indices[1:]
        if len(coeff_indices) == 0:
            raise ValueError("No DCT coefficients selected. Increase dct_keep_size or disable drop_dc.")

        self.patch_size = patch_size
        self.num_feature_levels = num_feature_levels
        self.dct_keep_size = dct_keep_size
        self.drop_dc = drop_dc
        self.num_kept_coeffs = len(coeff_indices)
        self.register_buffer("_dct_basis", _make_dct_basis(patch_size), persistent=False)
        self.register_buffer("_coeff_indices", torch.tensor(coeff_indices, dtype=torch.long), persistent=False)

        self.coeff_norm = nn.LayerNorm(self.num_kept_coeffs)
        self.patch_encoder = nn.Sequential(
            nn.Linear(self.num_kept_coeffs, branch_channels),
            nn.GELU(),
            nn.Linear(branch_channels, branch_channels),
            nn.GELU(),
        )
        self.level_projections = nn.ModuleList(
            [nn.Conv2d(branch_channels, 1, kernel_size=1) for _ in range(num_feature_levels)]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.patch_encoder:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.constant_(module.bias, 0)
        for projection in self.level_projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.constant_(projection.bias, 0)

    def _preprocess(self, images_denorm: torch.Tensor) -> torch.Tensor:
        if images_denorm.ndim != 4 or images_denorm.shape[1] != 3:
            raise ValueError(
                "SpectralPriorBranch expects denormalized RGB images with shape [B, 3, H, W], "
                f"got {tuple(images_denorm.shape)}."
            )

        # Luma keeps structural information while staying lightweight.
        grayscale = (
            0.2989 * images_denorm[:, 0:1]
            + 0.5870 * images_denorm[:, 1:2]
            + 0.1140 * images_denorm[:, 2:3]
        )
        mean = grayscale.mean(dim=(-2, -1), keepdim=True)
        std = grayscale.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (grayscale - mean) / std

    def _extract_patch_features(self, normalized_gray: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = normalized_gray.shape
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            normalized_gray = F.pad(normalized_gray, (0, pad_w, 0, pad_h), mode="replicate")

        padded_h = normalized_gray.shape[-2]
        padded_w = normalized_gray.shape[-1]
        grid_h = padded_h // self.patch_size
        grid_w = padded_w // self.patch_size

        patches = F.unfold(normalized_gray, kernel_size=self.patch_size, stride=self.patch_size)
        patches = patches.transpose(1, 2).reshape(-1, self.patch_size, self.patch_size)

        basis = self._dct_basis.to(device=normalized_gray.device, dtype=normalized_gray.dtype)
        dct_rows = torch.matmul(basis, patches)
        dct = torch.matmul(dct_rows, basis.transpose(0, 1))
        dct_flat = dct.reshape(-1, self.patch_size * self.patch_size)
        dct_selected = dct_flat.index_select(dim=1, index=self._coeff_indices)

        tokens = self.coeff_norm(dct_selected)
        tokens = self.patch_encoder(tokens)

        encoded = tokens.view(batch_size, grid_h * grid_w, -1)
        encoded = encoded.transpose(1, 2).reshape(batch_size, -1, grid_h, grid_w)
        return encoded

    def forward(
        self,
        images_denorm: torch.Tensor,
        target_features: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        normalized_gray = self._preprocess(images_denorm.float())
        patch_features = self._extract_patch_features(normalized_gray)

        priors: List[torch.Tensor] = []
        for level_idx, feature in enumerate(target_features):
            level_prior = self.level_projections[level_idx](patch_features)
            level_prior = F.interpolate(
                level_prior,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            priors.append(level_prior.to(dtype=feature.dtype))
        return priors


class SummitDINOModel(nn.Module):
    def __init__(
        self,
        summit_checkpoint_path: Optional[str] = None,
        backbone_variant: str = "auto",
        backbone_img_size: int = 448,
        in_chans: int = 3,
        neck_out_channels: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_queries: int = 300,
        num_feature_levels: int = 4,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        enc_n_points: int = 4,
        dec_n_points: int = 4,
        with_box_refine: bool = True,
        two_stage: bool = True,
        include_no_object: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone: SUMMITBackbone = build_summit_backbone(
            img_size=backbone_img_size,
            in_chans=in_chans,
            backbone_variant=backbone_variant,
            checkpoint_path=summit_checkpoint_path,
        )
        if summit_checkpoint_path:
            self.backbone.load_summit_checkpoint(summit_checkpoint_path)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.neck = SimpleFeaturePyramid(
            in_channels=self.backbone.out_channels,
            out_channels=neck_out_channels,
            num_feature_levels=num_feature_levels,
        )
        self.detector = DeformableDINOHead(
            in_channels_list=[neck_out_channels] * num_feature_levels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
            nheads=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_n_points=enc_n_points,
            dec_n_points=dec_n_points,
            with_box_refine=with_box_refine,
            two_stage=two_stage,
            include_no_object=include_no_object,
        )

    def init_backbone_from_checkpoint(self, checkpoint_path: str) -> None:
        self.backbone.load_summit_checkpoint(checkpoint_path)

    @staticmethod
    def _resize_masks(
        image_padding_mask: torch.Tensor, multi_scale_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        if image_padding_mask.dtype != torch.bool:
            image_padding_mask = image_padding_mask.to(torch.bool)
        masks = []
        for feature in multi_scale_features:
            resized = F.interpolate(
                image_padding_mask[:, None].float(),
                size=feature.shape[-2:],
                mode="nearest",
            )[:, 0].to(torch.bool)
            masks.append(resized)
        return masks

    def forward(
        self,
        images: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None,
        dn_queries: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        backbone_feature = self.backbone(images)
        multi_scale_features = self.neck(backbone_feature)
        masks = None
        if image_padding_mask is not None:
            masks = self._resize_masks(image_padding_mask, multi_scale_features)
        return self.detector(multi_scale_features, masks=masks, dn_queries=dn_queries)


def build_summit_dino_model(**kwargs) -> SummitDINOModel:
    return SummitDINOModel(**kwargs)


class DINOv2DINOModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2",
        backbone_checkpoint_path: Optional[str] = None,
        backbone_pretrained: bool = True,
        neck_out_channels: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_queries: int = 300,
        num_feature_levels: int = 4,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        enc_n_points: int = 4,
        dec_n_points: int = 4,
        with_box_refine: bool = True,
        two_stage: bool = True,
        include_no_object: bool = True,
        freeze_backbone: bool = False,
        use_spectral_prior: bool = True,
        spectral_fuse_level_indices: Tuple[int, ...] = (1, 2, 3),
        spectral_blur_kernel_size: int = 31,
        spectral_blur_sigma: float = 6.0,
        spectral_branch_channels: int = 64,
        spectral_patch_size: int = 8,
        spectral_dct_keep_size: int = 4,
        spectral_drop_dc: bool = False,
    ) -> None:
        super().__init__()
        _ = spectral_blur_kernel_size, spectral_blur_sigma
        self.backbone: DINOv2Backbone = build_dinov2_backbone(
            model_name=backbone_name,
            checkpoint_path=backbone_checkpoint_path,
            pretrained=backbone_pretrained,
        )

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.neck = SimpleFeaturePyramid(
            in_channels=self.backbone.out_channels,
            out_channels=neck_out_channels,
            num_feature_levels=num_feature_levels,
        )
        self.detector = DeformableDINOHead(
            in_channels_list=[neck_out_channels] * num_feature_levels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
            nheads=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            enc_n_points=enc_n_points,
            dec_n_points=dec_n_points,
            with_box_refine=with_box_refine,
            two_stage=two_stage,
            include_no_object=include_no_object,
        )
        self.use_spectral_prior = use_spectral_prior
        self.num_feature_levels = num_feature_levels
        self.spectral_fuse_level_indices = tuple(int(v) for v in spectral_fuse_level_indices)
        if len(set(self.spectral_fuse_level_indices)) != len(self.spectral_fuse_level_indices):
            raise ValueError(
                f"spectral_fuse_level_indices must not contain duplicates, got {self.spectral_fuse_level_indices}."
            )
        if self.use_spectral_prior and len(self.spectral_fuse_level_indices) == 0:
            raise ValueError("spectral_fuse_level_indices cannot be empty when use_spectral_prior=True.")
        for level_idx in self.spectral_fuse_level_indices:
            if level_idx < 0 or level_idx >= self.num_feature_levels:
                raise ValueError(
                    f"spectral_fuse_level_indices contains out-of-range level {level_idx}. "
                    f"Valid range is [0, {self.num_feature_levels - 1}]."
                )

        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.spectral_level_to_alpha_idx: Dict[int, int] = {
            level: alpha_idx for alpha_idx, level in enumerate(self.spectral_fuse_level_indices)
        }
        if self.use_spectral_prior:
            self.spectral_prior_branch = SpectralPriorBranch(
                num_feature_levels=self.num_feature_levels,
                branch_channels=spectral_branch_channels,
                patch_size=spectral_patch_size,
                dct_keep_size=spectral_dct_keep_size,
                drop_dc=spectral_drop_dc,
            )
            self.spectral_alphas = nn.Parameter(torch.zeros(len(self.spectral_fuse_level_indices)))
        else:
            self.spectral_prior_branch = None
            self.register_parameter("spectral_alphas", None)

    @staticmethod
    def _resize_masks(
        image_padding_mask: torch.Tensor, multi_scale_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        return SummitDINOModel._resize_masks(image_padding_mask, multi_scale_features)

    def forward(
        self,
        images: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None,
        dn_queries: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        backbone_feature = self.backbone(images)
        multi_scale_features = self.neck(backbone_feature)
        if self.use_spectral_prior and self.spectral_prior_branch is not None and self.spectral_alphas is not None:
            denorm_images = images * self._imagenet_std.to(images.dtype) + self._imagenet_mean.to(images.dtype)
            prior_maps = self.spectral_prior_branch(denorm_images, multi_scale_features)
            for level_idx in self.spectral_fuse_level_indices:
                alpha_idx = self.spectral_level_to_alpha_idx[level_idx]
                alpha = self.spectral_alphas[alpha_idx].view(1, 1, 1, 1)
                gate = torch.sigmoid(prior_maps[level_idx])
                multi_scale_features[level_idx] = multi_scale_features[level_idx] * (1.0 + alpha * gate)
        masks = None
        if image_padding_mask is not None:
            masks = self._resize_masks(image_padding_mask, multi_scale_features)
        return self.detector(multi_scale_features, masks=masks, dn_queries=dn_queries)


def build_dinov2_dino_model(**kwargs) -> DINOv2DINOModel:
    return DINOv2DINOModel(**kwargs)
