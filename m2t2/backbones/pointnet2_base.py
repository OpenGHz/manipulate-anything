# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch.nn as nn


class PointNet2Base(nn.Module):
    def __init__(self):
        super(PointNet2Base, self).__init__()
        self.SA_modules = nn.ModuleList()
        self.FP_modules = nn.ModuleList()

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = None
        if pc.shape[-1] > 3 and self.use_rgb:
            features = pc[..., 3:].transpose(1, 2).contiguous()
        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features, sample_ids = [xyz], [features], []
        for i in range(len(self.SA_modules)):
            li_xyz, _, li_features, sample_idx = self.SA_modules[i](
                l_xyz[i], l_features[i]
            )
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            if sample_idx[0] is not None:
                sample_ids.append(sample_idx[0])

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        l_features = {
            f'res{i}': feat for i, feat in enumerate(l_features)
            if feat is not None
        }
        l_xyz = {
            f'res{i}': xyz for i, xyz in enumerate(l_xyz) if xyz is not None
        }
        outputs = {
            'features': l_features,
            'context_pos': l_xyz,
            'sample_ids': sample_ids
        }
        return outputs
