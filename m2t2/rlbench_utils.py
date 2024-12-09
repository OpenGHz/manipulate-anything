from pyrep.const import RenderMode
from rlbench import CameraConfig, ObservationConfig
import numpy as np
import trimesh.transformations as tra


def create_obs_config(camera_names, camera_resolution):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=True,
        depth=True,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config


def pcd_rgb_within_bound(obs, cameras, bounds, channel_first=False):
    pcds, rgbs, masks = [], [], []
    for camera in cameras:
        pcd = obs[f'{camera}_point_cloud']
        rgb = obs[f'{camera}_rgb']
        if channel_first:
            pcd = pcd.transpose(1, 2, 0)
            rgb = rgb.transpose(1, 2, 0)
        pcds.append(pcd.reshape(-1, 3))
        rgbs.append(rgb.reshape(-1, 3))
        mask = obs[f'{camera}_mask']
        if np.issubdtype(mask.dtype, np.floating):
            mask = (mask[..., 0] * 255).astype(np.uint8)
        masks.append(mask.reshape(-1))
    pcd = np.concatenate(pcds)
    rgb = np.concatenate(rgbs)
    mask = np.concatenate(masks)
    within = (pcd[:, 0] > bounds[0]) & (pcd[:, 0] < bounds[3]) \
           & (pcd[:, 1] > bounds[1]) & (pcd[:, 1] < bounds[4]) \
           & (pcd[:, 2] > bounds[2]) & (pcd[:, 2] < bounds[5])
    return pcd[within], rgb[within], mask[within]


def gripper_pose_from_rlbench(pose, gripper_depth=0.1034):
    pose = pose @ tra.euler_matrix(0, 0, np.pi / 2)
    pose[:3, 3] -= gripper_depth * pose[:3, 2]
    return pose


def gripper_pose_to_rlbench(pose, gripper_depth=0.1034):
    pose_out = pose.copy()
    pose_out[:3, 3] += gripper_depth * pose[:3, 2]
    pose_out = pose_out @ tra.euler_matrix(0, 0, -np.pi / 2)
    return pose_out


def rotation_to_rlbench(pose):
    return np.roll(tra.quaternion_from_matrix(pose), -1)