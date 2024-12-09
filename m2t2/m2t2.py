# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn

from m2t2.action_decoder import ActionDecoder, infer_placements
from m2t2.backbones.build import build_backbone
from m2t2.contact_decoder import ContactDecoder


class M2T2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        object_encoder: nn.Module = None,
        grasp_mlp: nn.Module = None
    ):
        super(M2T2, self).__init__()
        self.backbone = backbone
        self.object_encoder = object_encoder
        self.transformer = transformer
        self.grasp_mlp = grasp_mlp

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args['backbone'] = build_backbone(cfg.scene_encoder)
        channels = args['backbone'].out_channels
        obj_channels = None
        if 'place' in cfg.contact_decoder.tasks:
            args['object_encoder'] = build_backbone(cfg.object_encoder)
            obj_channels = args['object_encoder'].out_channels
        args['transformer'] = ContactDecoder.from_config(
            cfg.contact_decoder, channels, obj_channels
        )
        if 'grasp' in cfg.contact_decoder.tasks:
            args['grasp_mlp'] = ActionDecoder.from_config(
                cfg.action_decoder, args['transformer']
            )
        return args

    def get_outputs(self, data):
        scene_features = self.backbone(data['inputs'])
        object_inputs = data['object_inputs']
        obj_features = {}
        if 'place' in self.transformer.tasks:
            obj_features = self.object_encoder(object_inputs)
        if 'task_is_place' in data:
            for key, val in obj_features['features'].items():
                obj_features['features'][key] = (
                    val * data['task_is_place'].view(
                        data['task_is_place'].shape[0], 1, 1, 1
                    )
                )
        embedding, outputs = self.transformer(scene_features, obj_features)

        losses = {}
        if 'place' in embedding and embedding['place'].shape[1] > 0:
            losses, stats = self.place_criterion(outputs, data)
            outputs[-1].update(stats)

        if 'grasp' in embedding and embedding['grasp'].shape[1] > 0:
            set_losses, outputs = self.set_criterion(outputs, data)
            losses.update(set_losses)
        else:
            outputs = outputs[-1]

        if self.grasp_mlp is not None:
            mask_features = scene_features['features'][
                self.transformer.mask_feature
            ]
            obj_embedding = [emb[idx] for emb, idx in zip(
                embedding['grasp'], outputs['matched_idx']
            )]
            grasp_outputs = self.grasp_mlp(
                data['points'], mask_features,
                outputs['matched_contact_masks'],
                obj_embedding, data['contact_masks']
            )
            outputs.update(grasp_outputs)
            contact_losses = self.grasp_criterion(outputs, data)
            losses.update(contact_losses)

        return outputs, losses

    def infer(self, data, cfg):
        scene_features = self.backbone(data['inputs'])
        obj_features = self.object_encoder(data['object_inputs'])
        if 'task_is_place' in data:
            for key in obj_features['features']:
                obj_features['features'][key] = (
                    obj_features['features'][key] * data['task_is_place'].view(
                        data['task_is_place'].shape[0], 1, 1
                    )
                )
        embedding, outputs = self.transformer(scene_features, obj_features)
        outputs = outputs[-1]

        if 'place' in embedding and embedding['place'].shape[1] > 0:
            cam_pose = data['cam_pose'] if cfg.cam_coord else None
            placement_outputs = infer_placements(
                data['points'], outputs['placement_masks'],
                data['object_points'], data['ee_pose'],
                cam_pose, cfg.mask_thresh, cfg.grid_res
            )
            outputs.update(placement_outputs)

        if 'objectness' in outputs:
            outputs['objectness'] = outputs['objectness'].sigmoid()
            object_ids = [
                torch.where(objectness > cfg.object_thresh)[0]
                for objectness in outputs['objectness']
            ]
            outputs['object_ids'] = object_ids

        for key in ['instance', 'contact']:
            if f'{key}_masks' in outputs:
                outputs[f'inferred_{key}_masks'] = [
                    mask[idx] for mask, idx in zip(
                        outputs[f'{key}_masks'], object_ids
                    )
                ]

        if 'grasp' in embedding and embedding['grasp'].shape[1] > 0:
            mask_features = scene_features['features'][
                self.transformer.mask_feature
            ]
            obj_embedding = [emb[idx] for emb, idx in zip(
                embedding['grasp'], object_ids
            )]
            grasp_outputs = self.grasp_mlp(
                data['points'], mask_features,
                outputs['inferred_contact_masks'], obj_embedding
            )
            outputs.update(grasp_outputs)

        return outputs
