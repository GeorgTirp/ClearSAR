import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        value: (bs, sum(hl*wl), n_heads, head_dim)
        value_spatial_shapes: (n_levels, 2) with (h, w)
        sampling_locations: (bs, len_q, n_heads, n_levels, n_points, 2), normalized in [0, 1]
        attention_weights: (bs, len_q, n_heads, n_levels, n_points)
    Returns:
        output: (bs, len_q, n_heads * head_dim)
    """
    bs, _, n_heads, head_dim = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape

    level_start_index = torch.cat(
        (
            value_spatial_shapes.new_zeros((1,)),
            value_spatial_shapes.prod(1).cumsum(0)[:-1],
        )
    )

    sampling_grids = 2.0 * sampling_locations - 1.0
    sampled_value_list = []

    for level in range(n_levels):
        h_l, w_l = value_spatial_shapes[level].tolist()
        start = int(level_start_index[level].item())
        end = start + h_l * w_l

        value_l = value[:, start:end].reshape(bs, h_l, w_l, n_heads, head_dim)
        value_l = value_l.permute(0, 3, 4, 1, 2).reshape(bs * n_heads, head_dim, h_l, w_l)

        sampling_grid_l = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4)
        sampling_grid_l = sampling_grid_l.reshape(bs * n_heads, len_q, n_points, 2)

        sampled_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_value_list.append(sampled_value_l)

    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_heads, 1, len_q, n_levels * n_points
    )
    sampled_value = torch.cat(sampled_value_list, dim=-1)
    output = (sampled_value * attention_weights).sum(-1)
    output = output.view(bs, n_heads * head_dim, len_q).transpose(1, 2).contiguous()
    return output


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (bs, len_q, d_model)
            reference_points: (bs, len_q, n_levels, 2 or 4)
            input_flatten: (bs, sum(hl*wl), d_model)
            input_spatial_shapes: (n_levels, 2)
            input_level_start_index: (n_levels,)
            input_padding_mask: (bs, sum(hl*wl)), True for padded positions
        """
        bs, len_q, _ = query.shape
        bs_value, len_in, _ = input_flatten.shape
        if bs_value != bs:
            raise ValueError("Batch size mismatch between query and input_flatten.")
        if int(input_spatial_shapes.prod(1).sum().item()) != len_in:
            raise ValueError("Spatial shapes do not match flattened input length.")

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(bs, len_in, self.n_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[:, 1], input_spatial_shapes[:, 0]], dim=-1
            )
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[
                None, None, None, :, None, :
            ]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + (
                sampling_offsets / float(self.n_points)
            ) * (reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 (xy) or 4 (xywh), "
                f"got {reference_points.shape[-1]}."
            )

        output = ms_deform_attn_core_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attention_weights,
        )
        output = self.output_proj(output)
        return output
