# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn.functional as F
import trimesh.transformations as tra

from m2t2.model_utils import MLP, repeat_new_axis


def double_split(tensor, chunks):
    tensor = list(tensor.split([sum(chunk) for chunk in chunks]))
    tensor = [
        list(elem.split([n for n in chunk]))
        for elem, chunk in zip(tensor, chunks)
    ]
    return tensor


def build_6d_grasp(
    contact_pt, contact_dir, approach_dir, offset, gripper_depth=0.1034
):
    grasp_tr = torch.stack([
        contact_dir,
        torch.cross(approach_dir, contact_dir),
        approach_dir,
        contact_pt + contact_dir * offset.unsqueeze(-1) / 2
        - gripper_depth * approach_dir
    ], axis=-1)
    last_row = torch.tensor([[0, 0, 0, 1]]).to(grasp_tr.device)
    if len(grasp_tr.shape) > 2:
        last_row = last_row * torch.ones(
            *grasp_tr.shape[:-2], 1, 4, device=grasp_tr.device
        )
    grasp_tr = torch.cat([grasp_tr, last_row], dim=-2)
    return grasp_tr


def build_6d_place(contact_pts, rot, offset, ee_pose):
    # Transformation order: first rotate gripper to grasp pose,
    # then add offset between gripper center and reference point,
    # then rotate around object center, finally translate to contact point.
    rot = rot @ ee_pose[..., :3, :3]
    trans = (contact_pts + offset).unsqueeze(-1)
    place_tr = torch.cat([rot, trans], axis=-1)
    last_row = torch.tensor([[0, 0, 0, 1]]).to(place_tr.device)
    if len(place_tr.shape) > 2:
        last_row = last_row * torch.ones(
            *place_tr.shape[:-2], 1, 4, device=place_tr.device
        )
    place_tr = torch.cat([place_tr, last_row], dim=-2)
    return place_tr


def compute_offset(obj_pts, ee_pose, rot, grid_res=0, cam_pose=None):
    # rot R is about object center o
    # offset is ee position e - target position t
    # R(e - o) - R(t - o) = -R(t - e)
    if cam_pose is not None:
        rot = cam_pose[:3, :3] @ rot
    obj_pts_stable = (obj_pts - ee_pose[:3, 3]) @ rot.transpose(-1, -2)
    if grid_res > 0:
        obj_pts_grid = (obj_pts_stable[..., :2] / grid_res).round()
        offset = obj_pts_stable.min(dim=0)[0]
        offset[:2] = obj_pts_grid.unique(dim=0).mean(dim=0) * grid_res
    else:
        offset = obj_pts_stable.mean(dim=0)
        offset[..., 2] = obj_pts_stable[..., 2].min(dim=1)[0]
    offset = -offset
    if cam_pose is not None:
        offset = offset @ cam_pose[:3, :3]
    return offset


def infer_placements(
    xyz, logits, obj_xyz, ee_poses, cam_poses, conf_thresh, grid_res
):
    """
    Args:
        xyz: [num_scenes, num_points, 3]
        logits: [num_scenes, num_rotations, num_points]
        obj_xyz: [num_scenes, num_object_points, 3]
        ee_poses: [num_scenes, 4, 4]
        cam_poses: [num_scenes, 4, 4]
        grid_res: float
    """
    rot_prompts = torch.stack([torch.from_numpy(
        tra.euler_matrix(0, 0, 2 * np.pi / logits.shape[1] * i)
    )[:3, :3].float() for i in range(logits.shape[1])]).to(xyz.device)
    rot_prompts = repeat_new_axis(rot_prompts, xyz.shape[1], dim=0)

    placements, confidence, contact_points = [], [], []
    for i, (pts, obj_pts, ee_pose, logit) in enumerate(zip(
        xyz, obj_xyz, ee_poses, logits
    )):
        conf = logit.T.sigmoid()
        mask = conf > conf_thresh
        num = mask.sum(dim=1)
        num = list(num[num > 0])
        contacts = repeat_new_axis(pts, mask.shape[1], dim=1)[mask]
        rot = rot_prompts[mask]
        offset = compute_offset(
            obj_pts, ee_pose, torch.eye(3).to(pts.device), grid_res
        )
        offsets = offset @ rot.transpose(1, 2)
        if cam_poses is not None:
            pts = pts @ cam_poses[i, :3, :3].T + cam_poses[i, :3, 3]
        place = build_6d_place(contacts, rot, offsets, ee_pose)
        placements.append(list(place.split(num)))
        confidence.append(list(conf[mask].split(num)))
        contact_points.append(pts[mask.any(dim=1)])
    outputs = {
        'placements': placements,
        'place_confidence': confidence,
        'place_contacts': contact_points,
    }
    return outputs


class ActionDecoder(torch.nn.Module):
    def __init__(
        self, feat_dim, use_embed, conf_thresh, max_num_pred,
        hidden_dim, num_layers, activation, offset_bins
    ):
        super(ActionDecoder, self).__init__()
        self.contact_dir_head = MLP(
            feat_dim, hidden_dim, 3, num_layers, activation
        )
        self.approach_dir_head = MLP(
            feat_dim, hidden_dim, 3, num_layers, activation
        )
        self.offset_head = MLP(
            feat_dim, hidden_dim, len(offset_bins) - 1,
            num_layers, activation
        )
        offset_bins = torch.tensor(offset_bins).float()
        self.offset_vals = (offset_bins[:-1] + offset_bins[1:]) / 2
        self.conf_thresh = conf_thresh
        self.max_num_pred = max_num_pred
        self.use_embed = use_embed

    @classmethod
    def from_config(cls, cfg, contact_decoder):
        args = {}
        args['feat_dim'] = contact_decoder.mask_dim
        args['use_embed'] = cfg.use_embed
        if cfg.use_embed:
            args['feat_dim'] += contact_decoder.embed_dim
        args['conf_thresh'] = cfg.conf_thresh
        args['max_num_pred'] = cfg.max_num_pred
        args['hidden_dim'] = cfg.hidden_dim
        args['num_layers'] = cfg.num_layers
        args['activation'] = cfg.activation
        args['offset_bins'] = cfg.offset_bins
        return cls(**args)

    def forward(self, xyz, mask_feats, logits, embedding, gt_masks=None):
        """
        Args:
            xyz: [num_scenes, num_points, 3]
            mask_feats: [num_scenes, feat_dim, num_points]
            logits: List of [num_objects, num_points]
            embedding: List of [num_objects, embed_dim]
            gt_masks: List of [num_objects, num_points]
        """
        mask_feats = mask_feats.moveaxis(1, -1) # [B, H, W, mask_dim]
        contacts, confidence, inputs, num_grasps = [], [], [], []
        total_grasps, num_objs = 0, 0
        for i, (pts, feat, emb, logit) in enumerate(
            zip(xyz, mask_feats, embedding, logits)
        ):
            conf_all = logit.sigmoid()
            mask = conf_all > self.conf_thresh
            if gt_masks is not None:
                mask = mask | (gt_masks[i] > 0)
            conf_list, num = [], []
            for e, m, conf in zip(emb, mask, conf_all):
                f, p, c = feat[m], pts[m], conf[m]
                if self.max_num_pred is not None:
                    perm = torch.randperm(f.shape[0])[self.max_num_pred]
                    perm = perm.to(f.device)
                    f, p, c = f[perm], p[perm], c[perm]
                if self.use_embed:
                    f = torch.cat([
                        f, repeat_new_axis(e, f.shape[0], dim=0)
                    ], dim=-1)
                contacts.append(p)
                inputs.append(f)
                conf_list.append(c)
                num.append(f.shape[0])
                total_grasps += f.shape[0]
            confidence.append(conf_list)
            num_grasps.append(num)
            num_objs += logit.shape[0]
        contacts = torch.cat(contacts)
        inputs = torch.cat(inputs)

        if gt_masks is not None:
            gt_inputs, total_gt_grasps = [], 0
            for feat, emb, mask in zip(mask_feats, embedding, gt_masks):
                for e, m in zip(emb, mask):
                    f = feat[m > 0]
                    if self.use_embed:
                        f = torch.cat([
                            f, repeat_new_axis(e, f.shape[0], 0)
                        ], dim=-1)
                    gt_inputs.append(f)
                    total_gt_grasps += f.shape[0]
            inputs = torch.cat([inputs, torch.cat(gt_inputs)])

        contact_dirs = F.normalize(self.contact_dir_head(inputs), dim=-1)
        approach_dirs = self.approach_dir_head(inputs)
        approach_dirs = F.normalize(
            approach_dirs - contact_dirs * (
                approach_dirs * contact_dirs
            ).sum(dim=-1, keepdim=True), dim=-1
        )
        offset_logits = self.offset_head(inputs)
        offsets_one_hot = F.one_hot(
            offset_logits.argmax(dim=-1), self.offset_vals.shape[0]
        )
        offsets = (
            offsets_one_hot.float() @ self.offset_vals.to(inputs.device)
        ).squeeze(-1)

        if gt_masks is not None:
            contact_dirs, outputs['contact_dirs'] = contact_dirs.split(
                [total_grasps, total_gt_grasps], dim=0
            )
            approach_dirs, outputs['approach_dirs'] = approach_dirs.split(
                [total_grasps, total_gt_grasps], dim=0
            )
            offsets = offsets[:total_grasps]
            outputs['offsets'] = offset_logits[total_grasps:]
        
        grasps = build_6d_grasp(contacts, contact_dirs, approach_dirs, offsets)
        grasps = double_split(grasps, num_grasps)
        contacts = double_split(contacts, num_grasps)
        outputs = {
            'grasps': grasps,
            'confidence': confidence,
            'contacts': contacts,
            'num_pred_grasps': torch.tensor(
                total_grasps / max(num_objs, 1), device=inputs.device
            )
        }
        outputs['grasp_pos_ratio'] = outputs['num_pred_grasps'] / xyz.shape[1]
        if gt_masks is not None:
            outputs['num_gt_grasps'] = torch.tensor(
                total_gt_grasps / max(num_objs, 1), device=inputs.device
            )
        return outputs
