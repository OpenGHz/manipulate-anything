# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Modified by Wentao Yuan from https://github.com/facebookresearch/Mask2Former
# /blob/main/mask2former/modeling/transformer_decoder
# /mask2former_transformer_decoder.py

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List

from m2t2.model_utils import MLP, get_activation_fn, repeat_new_axis


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, query, key, value, query_pos_enc, key_pos_enc, attn_mask=None
    ):
        output, attn = self.attn(
            query + query_pos_enc, key + key_pos_enc, value, attn_mask=attn_mask
        )
        return self.norm(query + output)


class FFNLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, activation="ReLU"):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            get_activation_fn(activation),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x + self.ff(x))


class PositionEncoding3D(nn.Module):
    """
    Generate sinusoidal positional encoding
    f(p) = (sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^L pi pi), cos(2^L pi p))
    """
    def __init__(self, enc_dim, scale=np.pi, temperature=10000):
        super(PositionEncoding3D, self).__init__()
        self.enc_dim = enc_dim
        self.freq = np.ceil(enc_dim / 6)
        self.scale = scale
        self.temperature = temperature

    def forward(self, pos):
        pos_min = pos.min(dim=-1, keepdim=True)[0]
        pos_max = pos.max(dim=-1, keepdim=True)[0]
        pos = ((pos - pos_min) / (pos_max - pos_min) - 0.5) * 2 * np.pi
        dim_t = torch.arange(self.freq, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (dim_t / self.freq)
        pos = pos[..., None] * self.scale / dim_t # (B, N, 3, F)
        pos = torch.stack([pos.sin(), pos.cos()], dim=-1).flatten(start_dim=2)
        pos = pos[..., :self.enc_dim].transpose(1, 2)
        return pos.detach()


def compute_attention_mask(attn_mask, nn_ids, context_size, num_heads):
    # resize attention mask to the size of context features
    if nn_ids is None:
        attn_mask = F.interpolate(attn_mask, context_size)
    else:
        j = 0
        while attn_mask.shape[-1] != context_size:
            # BxQxM -> BxQxMxK
            attn_mask = repeat_new_axis(attn_mask, nn_ids[j].shape[-1], dim=3)
            # BxNxK -> BxQxNxK
            idx = repeat_new_axis(nn_ids[j], attn_mask.shape[1], dim=1)
            # BxQxMxK -> BxQxNxK
            attn_mask = torch.gather(attn_mask, dim=2, index=idx.long())
            # BxQxNxK -> BxQxN
            attn_mask = attn_mask.max(dim=-1)[0]
            j += 1
    # If a BoolTensor is provided as attention mask,
    # positions with ``True`` are not allowed to attend.
    # We block attention where predicted mask logits < 0.
    attn_mask = (attn_mask < 0).bool()
    # [B, Q, N] -> [B, h, Q, N] -> [B*h, Q, N]
    attn_mask = repeat_new_axis(
        attn_mask, num_heads, dim=1
    ).flatten(start_dim=0, end_dim=1)
    # If attn mask is empty for any query, allow attention anywhere.
    attn_mask[
        torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])
    ] = False
    return attn_mask


class ContactDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_dim: int,
        tasks: List[str],
        num_object_queries: int,
        num_scene_queries: int,
        scene_in_features: List[str],
        scene_in_channels: List[int],
        mask_feature: str,
        mask_dim: int,
        place_feature: str,
        place_dim: int,
        num_layers: int,
        num_heads: int,
        use_attn_mask: bool,
        use_task_embed: bool,
        attn_mask_key: str,
        activation: str
    ):
        """
        Args:
            activation: activation function for the feedforward network
            embed_dim: transformer feature dimension
            feedforward_dim: hidden dimension of the feedforward network
            in_channels: feature dimensions of the input feature maps
            mask_dim: feature dimension of mask embedding
            num_layers: number of transformer decoder layers
            num_heads: number of attention heads
            num_queries: number of object queries
            instance_mask: predict instance masks
            contact_mask: predict contact masks
            contact_mask: predict contact parameters
            use_attn_mask: mask attention with downsampled instance mask
                           predicted by the previous layer
        """
        super(ContactDecoder, self).__init__()

        self.tasks = tasks
        self.num_object_queries = num_object_queries
        self.num_scene_queries = num_scene_queries
        num_queries = 0
        if 'instance' in tasks:
            num_queries += num_object_queries
        if 'grasp' in tasks:
            num_queries += num_object_queries
        if 'place' in tasks:
            num_queries += num_scene_queries
        # learnable grasp query features
        self.query_embed = nn.Embedding(
            num_object_queries + num_scene_queries, embed_dim
        )
        # learnable query p.e.
        self.query_pos_enc = nn.Embedding(num_queries, embed_dim)
        self.use_task_embed = use_task_embed
        if use_task_embed:
            # learnable task embedding
            self.task_embed = nn.Embedding(len(tasks), embed_dim)

        self.place_feature = place_feature
        if place_dim != embed_dim:
            self.place_embed_proj = nn.Linear(place_dim, embed_dim)
        else:
            self.place_embed_proj = nn.Identity()

        self.scene_in_features = scene_in_features
        self.num_scales = len(scene_in_features)
        # context scale embedding
        self.scale_embed = nn.Embedding(self.num_scales, embed_dim)
        # scene feature projection
        self.scene_feature_proj = nn.ModuleList([
            nn.Conv2d(channel, embed_dim, kernel_size=1)
            if channel != embed_dim else nn.Identity()
            for channel in scene_in_channels
        ])
        # context positional encoding
        self.pe_layer = PositionEncoding3D(embed_dim)

        # transformer decoder
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cross_attention_layers = nn.ModuleList()
        self.self_attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attention_layers.append(
                AttentionLayer(embed_dim, num_heads)
            )
            self.self_attention_layers.append(
                AttentionLayer(embed_dim, num_heads)
            )
            self.ffn_layers.append(
                FFNLayer(embed_dim, feedforward_dim, activation)
            )
        self.use_attn_mask = use_attn_mask
        self.attn_mask_key = attn_mask_key

        # prediction MLPs
        self.mask_feature = mask_feature
        self.mask_dim = mask_dim
        self.norm = nn.LayerNorm(embed_dim)
        if 'instance' in tasks or 'grasp' in tasks:
            self.object_head = nn.Linear(embed_dim, 1)
        for task in tasks:
            setattr(self, f'{task}_mask_head', MLP(
                embed_dim, embed_dim, mask_dim,
                num_layers=3, activation=activation
            ))

    @classmethod
    def from_config(cls, cfg, scene_channels, obj_channels):
        args = {}
        args["tasks"] = cfg.tasks
        args["mask_feature"] = cfg.mask_feature
        args["embed_dim"] = cfg.embed_dim
        args["feedforward_dim"] = cfg.feedforward_dim
        args["scene_in_features"] = cfg.in_features[::-1]
        args["scene_in_channels"] = [
            scene_channels[f] for f in cfg.in_features[::-1]
        ]
        args["num_object_queries"] = cfg.num_object_queries
        args["num_scene_queries"] = cfg.num_scene_queries
        args["mask_dim"] = scene_channels[cfg.mask_feature]
        args["place_feature"] = cfg.place_feature
        args["place_dim"] = obj_channels[cfg.place_feature]
        args["num_layers"] = cfg.num_layers
        args["num_heads"] = cfg.num_heads
        args["use_attn_mask"] = cfg.use_attn_mask
        args["use_task_embed"] = cfg.use_task_embed
        args["attn_mask_key"] = cfg.attn_mask_key
        args["activation"] = cfg.activation
        return cls(**args)

    def predict(self, embed, mask_features, num_queries):
        pred, attn_mask = {}, []
        for task, embed in zip(self.tasks, embed.split(num_queries)):
            embed = embed.transpose(0, 1)
            if task == 'instance':
                pred['objectness'] = self.object_head(embed).squeeze(-1)
                pred['instance_masks'] = torch.einsum(
                    "bqc,bcn->bqn", self.instance_mask_head(embed), mask_features
                )
                attn_mask.append(pred['instance_masks'])
            if task == 'grasp':
                if 'instance' not in self.tasks:
                    pred['objectness'] = self.object_head(embed).squeeze(-1)
                pred['contact_masks'] = torch.einsum(
                    "bqc,bcn->bqn", self.grasp_mask_head(embed), mask_features
                )
                attn_mask.append(pred['contact_masks'])
            if task == 'place':
                pred['placement_masks'] = torch.einsum(
                    "bqc,bcn->bqn", self.place_mask_head(embed), mask_features
                )
                attn_mask.append(pred['placement_masks'])
        attn_mask = torch.cat(attn_mask, dim=1).detach()
        return pred, attn_mask

    def construct_context(self, features, feature_keys, feature_proj):
        context = [features['features'][f] for f in feature_keys]
        pos_encs, context_sizes = [], []
        for i, f in enumerate(feature_keys):
            pos_enc = self.pe_layer(features['context_pos'][f])
            context_sizes.append(context[i].shape[-1])
            pos_enc = pos_enc.flatten(start_dim=2).permute(2, 0, 1)
            pos_encs.append(pos_enc)
            context[i] = feature_proj[i](context[i].unsqueeze(-1)).squeeze(-1)
            context[i] = context[i] + self.scale_embed.weight[i].unsqueeze(1)
            # NxCxHW -> HWxNxC
            context[i] = context[i].permute(2, 0, 1)
        return context, pos_encs, context_sizes

    def forward(self, scene_features, obj_features):
        """
        Args:
            scene_features: a dict containing multi-scale feature maps 
                            from scene point cloud
            obj_features: a dict containing multi-scale feature maps
                          from point cloud of object to be placed
        """
        context, pos_encs, context_sizes = self.construct_context(
            scene_features, self.scene_in_features, self.scene_feature_proj
        )
        mask_feat = scene_features['features'][self.mask_feature]

        object_embed, scene_embed = self.query_embed.weight.split(
            [self.num_object_queries, self.num_scene_queries]
        )
        embed = []
        for i, task in enumerate(self.tasks):
            if task == 'place':
                place_prompts = obj_features['features'][self.place_feature]
                place_prompts = place_prompts.max(dim=-1)[0]
                place_prompts = self.place_embed_proj(place_prompts)
                if len(place_prompts.shape) < 3:
                    if self.use_task_embed:
                        scene_embed = scene_embed + self.task_embed.weight[i]
                    embed.append(
                        scene_embed.unsqueeze(1) + place_prompts.unsqueeze(0)
                    )
                else:
                    place_prompts = place_prompts.transpose(0, 1)
                    if self.use_task_embed:
                        place_prompts = place_prompts + self.task_embed.weight[i]
                    embed.append(place_prompts)
            else:
                if self.use_task_embed:
                    object_embed = object_embed + self.task_embed.weight[i]
                embed.append(repeat_new_axis(
                    object_embed, mask_feat.shape[0], dim=1
                ))
        num_queries = [emb.shape[0] for emb in embed]
        embed = torch.cat(embed)
        query_pos_enc = repeat_new_axis(
            self.query_pos_enc.weight, mask_feat.shape[0], dim=1
        )
        if 'place' in self.tasks and len(place_prompts.shape) >= 3:
            num_rotations = place_prompts.shape[0]
            query_pos_enc = torch.cat([
                query_pos_enc[:-1], repeat_new_axis(
                    query_pos_enc[-1], num_rotations, dim=0
                )
            ])

        # initial prediction with learnable query features only (no context)
        embed = self.norm(embed)
        prediction, attn_mask = self.predict(embed, mask_feat, num_queries)
        predictions = [prediction]

        for i in range(self.num_layers):
            j = i % self.num_scales
            if self.use_attn_mask:
                attn_mask = compute_attention_mask(
                    attn_mask, scene_features['sample_ids'],
                    context_sizes[j], self.num_heads
                )
            else:
                attn_mask = None
            context_feat = context[j]
            key_pos_enc = pos_encs[j]
            embed = self.cross_attention_layers[i](
                embed, context_feat, context_feat + key_pos_enc,
                query_pos_enc, key_pos_enc, attn_mask
            )
            embed = self.self_attention_layers[i](
                embed, embed, embed + query_pos_enc,
                query_pos_enc, query_pos_enc
            )
            embed = self.ffn_layers[i](embed)

            prediction, attn_mask = self.predict(embed, mask_feat, num_queries)
            predictions.append(prediction)

        embedding = {}
        embeds = embed.split(num_queries)
        if 'place' in self.tasks:
            embedding['place'] = embeds[-1].transpose(0, 1)
            if 'grasp' in self.tasks:
                embedding['grasp'] = embeds[-2].transpose(0, 1)
        elif 'grasp' in self.tasks:
            embedding['grasp'] = embeds[-1].transpose(0, 1)
        return embedding, predictions
