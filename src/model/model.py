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


def _make_gaussian_kernel2d(kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Gaussian kernel size must be positive odd, got {kernel_size}.")
    if sigma <= 0:
        raise ValueError(f"Gaussian sigma must be positive, got {sigma}.")

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, kernel_size, kernel_size)


class SpectralPriorBranch(nn.Module):
    def __init__(
        self,
        num_feature_levels: int,
        branch_channels: int = 64,
        blur_kernel_size: int = 31,
        blur_sigma: float = 6.0,
    ) -> None:
        super().__init__()
        if num_feature_levels <= 0:
            raise ValueError(f"num_feature_levels must be > 0, got {num_feature_levels}.")
        if branch_channels <= 0:
            raise ValueError(f"branch_channels must be > 0, got {branch_channels}.")

        self.num_feature_levels = num_feature_levels
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.register_buffer(
            "_gaussian_kernel",
            _make_gaussian_kernel2d(kernel_size=blur_kernel_size, sigma=blur_sigma),
            persistent=False,
        )

        self.prior_extractor = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.level_projections = nn.ModuleList(
            [nn.Conv2d(branch_channels, 1, kernel_size=1) for _ in range(num_feature_levels)]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.prior_extractor:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
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

        # Luma conversion keeps structural intensity for spectral extraction.
        grayscale = (
            0.2989 * images_denorm[:, 0:1]
            + 0.5870 * images_denorm[:, 1:2]
            + 0.1140 * images_denorm[:, 2:3]
        )

        kernel = self._gaussian_kernel.to(device=images_denorm.device, dtype=images_denorm.dtype)
        blurred = F.conv2d(grayscale, kernel, padding=self.blur_kernel_size // 2)
        whitened = grayscale - blurred

        mean = whitened.mean(dim=(-2, -1), keepdim=True)
        std = whitened.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (whitened - mean) / std

    def _log_spectrum(self, whitened: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(whitened, dim=(-2, -1))
        magnitude = spectrum.abs()
        centered = torch.fft.fftshift(magnitude, dim=(-2, -1))
        log_spectrum = torch.log1p(centered)

        mean = log_spectrum.mean(dim=(-2, -1), keepdim=True)
        std = log_spectrum.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return (log_spectrum - mean) / std

    def forward(
        self,
        images_denorm: torch.Tensor,
        target_features: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        # Keep FFT path in float32 for numerical stability and broader device support.
        whitened = self._preprocess(images_denorm.float())
        spectral_map = self._log_spectrum(whitened)
        prior_features = self.prior_extractor(spectral_map)

        priors: List[torch.Tensor] = []
        for level_idx, feature in enumerate(target_features):
            level_prior = self.level_projections[level_idx](prior_features)
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
    ) -> None:
        super().__init__()
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
                blur_kernel_size=spectral_blur_kernel_size,
                blur_sigma=spectral_blur_sigma,
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
