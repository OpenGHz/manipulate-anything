# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch


def sample_points(xyz, num_points):
    if xyz.shape[0] != 0:
        num_replica = num_points // xyz.shape[0]
    else:
        # Handle the case when xyz.shape[0] is zero
        # For example, you can set num_replica to a default value
        num_replica = 0
    num_remain = num_points % xyz.shape[0]
    pt_idx = torch.randperm(xyz.shape[0])
    pt_idx = torch.cat(
        [pt_idx for _ in range(num_replica)] + [pt_idx[:num_remain]]
    )
    return pt_idx
