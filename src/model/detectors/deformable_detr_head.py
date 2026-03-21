import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import MSDeformAttn


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def gen_sineembed_for_position(pos_tensor: torch.Tensor, num_pos_feats: int) -> torch.Tensor:
    """
    Args:
        pos_tensor: (bs, nq, 2) normalized center points in [0, 1]
    Returns:
        (bs, nq, 2 * num_pos_feats)
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale

    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[idx], dims[idx + 1]) for idx in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = F.relu(x)
        return x


class SinePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float = 2 * math.pi,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        not_mask = (~mask).float()
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ) -> None:
        super().__init__()
        self.self_attn = MSDeformAttn(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: torch.Tensor) -> torch.Tensor:
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src: torch.Tensor,
        pos: Optional[torch.Tensor],
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(
        spatial_shapes: torch.Tensor, valid_ratios: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        reference_points_list = []
        for level, (h_l, w_l) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h_l - 0.5, int(h_l), dtype=torch.float32, device=device),
                torch.linspace(0.5, w_l - 0.5, int(w_l), dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * h_l)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * w_l)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformAttn(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: Optional[torch.Tensor],
        reference_points: torch.Tensor,
        src: torch.Tensor,
        src_spatial_shapes: torch.Tensor,
        src_level_start_index: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            src_level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableDINOHead(nn.Module):
    """
    Canonical Deformable-DINO style detector head:
    - Multi-scale deformable attention encoder
    - Two-stage query initialization from encoder proposals (optional)
    - Deformable decoder with iterative box refinement
    - Denoising-query hooks
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
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
    ) -> None:
        super().__init__()
        if len(in_channels_list) != num_feature_levels:
            raise ValueError(
                f"Expected {num_feature_levels} feature levels in in_channels_list, "
                f"got {len(in_channels_list)}."
            )

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels
        self.num_decoder_layers = num_decoder_layers
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.include_no_object = include_no_object
        self.class_dim = num_classes + (1 if include_no_object else 0)

        norm_groups = math.gcd(hidden_dim, 32)
        norm_groups = 1 if norm_groups == 0 else norm_groups
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(norm_groups, hidden_dim),
                )
                for in_channels in in_channels_list
            ]
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, hidden_dim))
        self.position_embedding = SinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=hidden_dim,
            d_ffn=dim_feedforward,
            dropout=dropout,
            n_levels=num_feature_levels,
            n_heads=nheads,
            n_points=enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=hidden_dim,
            d_ffn=dim_feedforward,
            dropout=dropout,
            n_levels=num_feature_levels,
            n_heads=nheads,
            n_points=dec_n_points,
        )
        self.decoder_layers = _get_clones(decoder_layer, num_decoder_layers)

        self.query_scale = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        if not two_stage:
            self.refpoint_embed = nn.Embedding(num_queries, 4)

        self.class_embed = _get_clones(nn.Linear(hidden_dim, self.class_dim), num_decoder_layers)
        self.bbox_embed = _get_clones(MLP(hidden_dim, hidden_dim, 4, 3), num_decoder_layers)

        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)
            self.enc_class_embed = nn.Linear(hidden_dim, self.class_dim)
            self.enc_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.constant_(proj[0].bias, 0)
            nn.init.constant_(proj[1].weight, 1)
            nn.init.constant_(proj[1].bias, 0)

        nn.init.normal_(self.level_embed)
        nn.init.normal_(self.tgt_embed.weight)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for cls_layer in self.class_embed:
            nn.init.constant_(cls_layer.bias, bias_value)

        for box_layer in self.bbox_embed:
            nn.init.constant_(box_layer.layers[-1].weight, 0)
            nn.init.constant_(box_layer.layers[-1].bias, 0)
            box_layer.layers[-1].bias.data[2:] = -2.0

        if self.two_stage:
            nn.init.xavier_uniform_(self.enc_output.weight)
            nn.init.constant_(self.enc_output.bias, 0)
            nn.init.constant_(self.enc_class_embed.bias, bias_value)
            nn.init.constant_(self.enc_bbox_embed.layers[-1].weight, 0)
            nn.init.constant_(self.enc_bbox_embed.layers[-1].bias, 0)
        else:
            nn.init.uniform_(self.refpoint_embed.weight[:, :2], 0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            nn.init.constant_(self.refpoint_embed.weight[:, 2:], 0)

    @staticmethod
    def _get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], dim=1)
        valid_w = torch.sum(~mask[:, 0, :], dim=1)
        valid_ratio_h = valid_h.float() / h
        valid_ratio_w = valid_w.float() / w
        return torch.stack([valid_ratio_w, valid_ratio_h], dim=-1)

    def _flatten_multi_level_inputs(
        self,
        multi_scale_features: Sequence[torch.Tensor],
        masks: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(multi_scale_features) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} feature maps, got {len(multi_scale_features)}."
            )
        if masks is not None and len(masks) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} masks, got {len(masks)}."
            )

        src_flatten: List[torch.Tensor] = []
        mask_flatten: List[torch.Tensor] = []
        pos_flatten: List[torch.Tensor] = []
        spatial_shapes: List[Tuple[int, int]] = []
        valid_ratios: List[torch.Tensor] = []

        for level, feat in enumerate(multi_scale_features):
            src = self.input_proj[level](feat)
            bs, _, h, w = src.shape
            spatial_shapes.append((h, w))

            if masks is None or masks[level] is None:
                mask = torch.zeros((bs, h, w), dtype=torch.bool, device=src.device)
            else:
                mask = masks[level].to(device=src.device, dtype=torch.bool)
                if mask.shape[-2:] != (h, w):
                    mask = F.interpolate(mask[:, None].float(), size=(h, w), mode="nearest").to(torch.bool)[:, 0]

            pos = self.position_embedding(mask) + self.level_embed[level].view(1, -1, 1, 1)

            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_flatten.append(pos.flatten(2).transpose(1, 2))
            valid_ratios.append(self._get_valid_ratio(mask))

        src_flatten_t = torch.cat(src_flatten, dim=1)
        mask_flatten_t = torch.cat(mask_flatten, dim=1)
        pos_flatten_t = torch.cat(pos_flatten, dim=1)
        spatial_shapes_t = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten_t.device)
        level_start_index = torch.cat(
            (
                spatial_shapes_t.new_zeros((1,)),
                spatial_shapes_t.prod(1).cumsum(0)[:-1],
            )
        )
        valid_ratios_t = torch.stack(valid_ratios, dim=1)

        return (
            src_flatten_t,
            mask_flatten_t,
            pos_flatten_t,
            spatial_shapes_t,
            level_start_index,
            valid_ratios_t,
        )

    def _gen_encoder_output_proposals(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, _, c = memory.shape
        proposals = []
        cur = 0
        for level, (h_l, w_l) in enumerate(spatial_shapes):
            h_l = int(h_l.item())
            w_l = int(w_l.item())
            mask_flatten = memory_padding_mask[:, cur : cur + h_l * w_l].view(bs, h_l, w_l, 1)

            valid_h = torch.sum(~mask_flatten[:, :, 0, 0], 1)
            valid_w = torch.sum(~mask_flatten[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, h_l - 1, h_l, dtype=torch.float32, device=memory.device),
                torch.linspace(0, w_l - 1, w_l, dtype=torch.float32, device=memory.device),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), dim=-1)

            scale = torch.stack((valid_w, valid_h), dim=1).view(bs, 1, 1, 2).float()
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * (0.05 * (2.0 ** level))
            proposal = torch.cat((grid, wh), dim=-1).view(bs, -1, 4)
            proposals.append(proposal)
            cur += h_l * w_l

        output_proposals = torch.cat(proposals, dim=1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals_unact = inverse_sigmoid(output_proposals)
        output_proposals_unact = output_proposals_unact.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals_unact = output_proposals_unact.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        output_memory = memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
        output_memory = output_memory.masked_fill(~output_proposals_valid, 0.0)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals_unact

    @staticmethod
    def _set_aux_loss(
        outputs_class: List[torch.Tensor],
        outputs_coord: List[torch.Tensor],
        dn_num: int = 0,
    ) -> List[Dict[str, torch.Tensor]]:
        aux_outputs = []
        for cls, box in zip(outputs_class[:-1], outputs_coord[:-1]):
            if dn_num > 0:
                aux_outputs.append(
                    {
                        "pred_logits": cls[:, dn_num:],
                        "pred_boxes": box[:, dn_num:],
                        "dn_pred_logits": cls[:, :dn_num],
                        "dn_pred_boxes": box[:, :dn_num],
                    }
                )
            else:
                aux_outputs.append({"pred_logits": cls, "pred_boxes": box})
        return aux_outputs

    def forward(
        self,
        multi_scale_features: Sequence[torch.Tensor],
        masks: Optional[Sequence[torch.Tensor]] = None,
        dn_queries: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        (
            src_flatten,
            mask_flatten,
            pos_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        ) = self._flatten_multi_level_inputs(multi_scale_features, masks)

        memory = self.encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            pos=pos_flatten,
            padding_mask=mask_flatten,
        )

        bs = memory.shape[0]
        enc_outputs_class = None
        enc_outputs_coord_unact = None

        if self.two_stage:
            output_memory, output_proposals = self._gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            enc_outputs_class = self.enc_class_embed(output_memory)
            enc_outputs_coord_unact = self.enc_bbox_embed(output_memory) + output_proposals

            if self.num_classes > 0:
                enc_score = enc_outputs_class[..., : self.num_classes].max(dim=-1).values
            else:
                enc_score = enc_outputs_class.max(dim=-1).values
            valid_proposals = torch.isfinite(output_proposals).all(dim=-1)
            enc_score = enc_score.masked_fill(~valid_proposals, float("-inf"))

            num_queries = min(self.num_queries, enc_score.shape[1])
            if num_queries < self.num_queries:
                raise ValueError(
                    f"num_queries ({self.num_queries}) is larger than available encoder tokens "
                    f"({enc_score.shape[1]})."
                )

            topk_indices = torch.topk(enc_score, num_queries, dim=1).indices
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
            )
            topk_memory = torch.gather(
                output_memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
            )
            tgt = topk_memory + self.tgt_embed.weight.unsqueeze(0)
            reference_points = topk_coords_unact.sigmoid()
        else:
            tgt = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            reference_points = self.refpoint_embed.weight.unsqueeze(0).repeat(bs, 1, 1).sigmoid()

        dn_num = 0
        self_attn_mask = None
        if dn_queries is not None:
            dn_tgt = dn_queries.get("tgt")
            dn_refpoints = dn_queries.get("refpoints")
            if dn_tgt is None or dn_refpoints is None:
                raise ValueError("dn_queries must contain both 'tgt' and 'refpoints'.")
            if dn_tgt.shape[0] != bs or dn_refpoints.shape[0] != bs:
                raise ValueError("dn_queries batch size must match feature batch size.")
            dn_num = dn_tgt.shape[1]
            tgt = torch.cat([dn_tgt, tgt], dim=1)
            reference_points = torch.cat([dn_refpoints, reference_points], dim=1)
            self_attn_mask = dn_queries.get("attn_mask")
            if self_attn_mask is not None:
                self_attn_mask = self_attn_mask.to(device=tgt.device)

        outputs_class: List[torch.Tensor] = []
        outputs_coord: List[torch.Tensor] = []

        output = tgt
        for layer_idx, layer in enumerate(self.decoder_layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat(
                    [valid_ratios, valid_ratios], dim=-1
                )[:, None]
            else:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = gen_sineembed_for_position(
                reference_points[..., :2], num_pos_feats=self.hidden_dim // 2
            )
            query_pos = self.ref_point_head(query_sine_embed)
            if layer_idx > 0:
                query_pos = query_pos * self.query_scale(output)

            output = layer(
                tgt=output,
                query_pos=query_pos,
                reference_points=reference_points_input,
                src=memory,
                src_spatial_shapes=spatial_shapes,
                src_level_start_index=level_start_index,
                src_padding_mask=mask_flatten,
                self_attn_mask=self_attn_mask,
            )

            cls_out = self.class_embed[layer_idx](output)
            box_delta = self.bbox_embed[layer_idx](output)
            box_out = (box_delta + inverse_sigmoid(reference_points)).sigmoid()

            outputs_class.append(cls_out)
            outputs_coord.append(box_out)

            if self.with_box_refine:
                reference_points = box_out.detach()

        pred_logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]

        out: Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]] | Dict[str, torch.Tensor]] = {
            "pred_logits": pred_logits[:, dn_num:] if dn_num > 0 else pred_logits,
            "pred_boxes": pred_boxes[:, dn_num:] if dn_num > 0 else pred_boxes,
            "aux_outputs": self._set_aux_loss(outputs_class, outputs_coord, dn_num=dn_num),
        }

        if dn_num > 0:
            out["dn_pred_logits"] = pred_logits[:, :dn_num]
            out["dn_pred_boxes"] = pred_boxes[:, :dn_num]

        if enc_outputs_class is not None and enc_outputs_coord_unact is not None:
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord_unact.sigmoid(),
            }

        return out  # type: ignore[return-value]


# Backward-compatible name.
DeformableDETRHead = DeformableDINOHead
