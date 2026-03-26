from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn


DINOV2_BACKBONE_CHOICES = (
    "vit_small_patch14_dinov2",
    "vit_base_patch14_dinov2",
    "vit_large_patch14_dinov2",
    "vit_giant_patch14_dinov2",
)

DINOV2_PATCH_SIZE = 14


class DINOv2Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        checkpoint_path: Optional[str] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in DINOV2_BACKBONE_CHOICES:
            raise ValueError(
                f"Unsupported DINOv2 backbone '{model_name}'. "
                f"Expected one of: {', '.join(DINOV2_BACKBONE_CHOICES)}"
            )

        create_kwargs = {
            "features_only": True,
            "out_indices": (-1,),
            "dynamic_img_size": True,
        }
        if checkpoint_path is not None:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_file}")
            create_kwargs["checkpoint_path"] = str(checkpoint_file)
            load_pretrained = False
        else:
            load_pretrained = pretrained

        try:
            self.model = timm.create_model(
                model_name,
                pretrained=load_pretrained,
                **create_kwargs,
            )
        except Exception as exc:
            if checkpoint_path is None and pretrained:
                raise RuntimeError(
                    f"Failed to load pretrained DINOv2 weights for '{model_name}'. "
                    "Provide --backbone-checkpoint with a local weights path or "
                    "ensure network/cache access for timm pretrained weights."
                ) from exc
            raise RuntimeError(f"Failed to initialize DINOv2 backbone '{model_name}'.") from exc

        channels = [int(v) for v in self.model.feature_info.channels()]
        reductions = [int(v) for v in self.model.feature_info.reduction()]
        if len(channels) != 1 or len(reductions) != 1:
            raise RuntimeError(
                f"Expected one feature map from DINOv2 features_only model, got "
                f"{len(channels)} channels entries and {len(reductions)} reduction entries."
            )
        self._out_channels = channels[0]
        self._stride = reductions[0]

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def patch_size(self) -> int:
        return DINOV2_PATCH_SIZE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        if not isinstance(features, (list, tuple)) or len(features) != 1:
            raise RuntimeError(f"Expected a single feature map, got type={type(features).__name__}")
        return features[0]


def build_dinov2_backbone(
    model_name: str = "vit_base_patch14_dinov2",
    checkpoint_path: Optional[str] = None,
    pretrained: bool = True,
) -> DINOv2Backbone:
    return DINOv2Backbone(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        pretrained=pretrained,
    )
