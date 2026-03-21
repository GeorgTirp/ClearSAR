from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SUMMITBackbone, build_summit_backbone
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
