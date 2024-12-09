from functools import partial
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from omegaconf import DictConfig
from scipy.spatial import KDTree
from rlbench.utils import get_stored_demos
from time import time
import hydra
import numpy as np
import torch

from m2t2.dataset_utils import sample_points
from m2t2.meshcat_utils import (
    create_visualizer, make_frame,
    visualize_grasp, visualize_pointcloud
)
from m2t2.model_utils import to_cpu, to_gpu, load_control_points
from m2t2.m2t2 import M2T2
from m2t2.rlbench_utils import (
    create_obs_config, pcd_rgb_within_bound, gripper_pose_from_rlbench
)


def select_object(cfg, demo, model, fig, event):
    x, y = int(event.xdata), int(event.ydata)
    circle = Circle((x, y), radius=2, color='r', fill=True, alpha=0.5)
    plt.gca().add_patch(circle)
    plt.draw()
    click_pt = torch.from_numpy(
        demo[f'{cfg.eval.ui_camera}_point_cloud'][y, x]
    ).float()

    pcd_raw, rgb_raw, mask_raw = pcd_rgb_within_bound(
        demo, cfg.rlbench.cameras, cfg.rlbench.scene_bounds
    )
    pt_idx = sample_points(pcd_raw, cfg.data.num_points)
    pcd, rgb = pcd_raw[pt_idx], rgb_raw[pt_idx]
    vis = create_visualizer()
    visualize_pointcloud(vis, 'scene', pcd, rgb, size=cfg.meshcat.point_size)

    if cfg.eval.task == 'pick':
        is_pick = torch.ones(1)
        is_place = torch.zeros(1)
        obj_pts = torch.rand(cfg.data.num_object_points, 3) - 0.5
        ee_pose = torch.eye(4)
    elif cfg.eval.task == 'place':
        is_pick = torch.zeros(1)
        is_place = torch.ones(1)
        obj_id = demo[f'{cfg.eval.ui_camera}_mask'][y, x]
        obj_pts = pcd_raw[mask_raw == obj_id]
        obj_pt_idx = sample_points(obj_pts, cfg.data.num_object_points)
        obj_pts = obj_pts[obj_pt_idx]
        visualize_pointcloud(
            vis, 'object', obj_pts, [255, 0, 0], size=cfg.meshcat.point_size
        )
        ee_pose = gripper_pose_from_rlbench(demo['gripper_matrix'])
        inv_ee_pose = np.linalg.inv(ee_pose)
        obj_pts_ee = obj_pts @ inv_ee_pose[:3, :3].T + inv_ee_pose[:3, 3]

    inputs = {
        'inputs': torch.from_numpy(
            select_region()- pcd_raw.mean(axis=0)
        ).unsqueeze(0).float(),
        'points': torch.from_numpy(pcd).unsqueeze(0).float(),
        'task_is_pick': is_pick,
        'task_is_place': is_place,
        'object_inputs': torch.from_numpy(
            obj_pts_ee - obj_pts_ee.mean(axis=0)
        ).float().unsqueeze(0),
        'object_points': [torch.from_numpy(obj_pts).float()],
        'ee_pose': torch.from_numpy(ee_pose).float().unsqueeze(0)
    }
    to_gpu(inputs)

    start = time()
    outputs = model.infer(inputs, cfg.eval)
    print('Inference time:', time() - start, 'seconds')
    to_cpu(outputs)

    if cfg.eval.task == 'pick':
        dists = torch.stack([
            (contacts - click_pt).norm(dim=1).mean()
            for contacts in outputs['contacts'][0]
        ]).nan_to_num(float('inf'))
        obj_id = dists.argmin()
        objectness = outputs["objectness"][0][
            outputs["object_ids"][0][obj_id]
        ].item()
        contacts = outputs['contacts'][0][obj_id].numpy()
        confidence = outputs['confidence'][0][obj_id].numpy()
        grasps = outputs['grasps'][0][obj_id]
        print('objectness', objectness, grasps.shape[0], 'grasps')
        if grasps.shape[0] > 0:
            visualize_pointcloud(
                vis, f"contacts", contacts, [0, 255, 0],
                size=cfg.meshcat.point_size * 2
            )
            pose_gen = fps_generator(grasps, confidence.argmax())
            for i, (grasp, idx) in enumerate(pose_gen):
                visualize_grasp(
                    vis, 'grasp', grasp.numpy(), [0, 255, 0],
                    linewidth=cfg.meshcat.line_width
                )
                input(f'Confidence {confidence[idx].item()}')
                if i >= cfg.eval.num_samples - 1:
                    break
        else:
            print('No grasp')
        plt.close(fig)
    elif cfg.eval.task == 'place':
        make_frame(vis, 'ee_pose', T=ee_pose)
        fig = plt.figure()
        plt.imshow(demo[f'{cfg.eval.ui_camera}_rgb'])
        fig.canvas.mpl_connect(
            'button_press_event', partial(
                select_region, cfg, demo, outputs, obj_pts_ee, vis, fig, []
            )
        )
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def select_region(cfg, demo, outputs, obj_pts, vis, fig, clicked_pts, event):
    x, y = int(event.xdata), int(event.ydata)
    clicked_pts.append([x, y])
    if len(clicked_pts) == 2:
        clicked_pts = np.array(clicked_pts)
        x0, y0 = clicked_pts.min(axis=0)
        x1, y1 = clicked_pts.max(axis=0)
        rect = Rectangle(
            (x0, y0), x1 - x0, y1 - y0, color='g', fill=True, alpha=0.5
        )
        ax = fig.get_axes()[0]
        ax.add_patch(rect)
        plt.draw()
        region = demo[f'{cfg.eval.ui_camera}_point_cloud'][y0:y1, x0:x1][
            demo[f'{cfg.eval.ui_camera}_mask'][y0:y1, x0:x1] > 0
        ]
        visualize_pointcloud(
            vis, "placement_region", region,
            [0, 0, 255], size=cfg.meshcat.point_size
        )
        tree = KDTree(region)
        contacts = outputs['place_contacts'][0].numpy()
        dists, _ = tree.query(contacts)
        matched = np.where(dists < 0.02)[0]
        visualize_pointcloud(
            vis, "contacts", contacts[matched], [0, 255, 0],
            size=cfg.meshcat.point_size
        )

        if len(matched) > 0:
            confidence, placements = [], []
            for i in matched:
                confidence.append(outputs['place_confidence'][0][i])
                placements.append(outputs['placements'][0][i])
            confidence = torch.cat(confidence)
            placements = torch.cat(placements)
            pose_gen = center_generator(placements, place_contacts)
            for i, (placement, idx) in enumerate(pose_gen):
                placement = placement.numpy()
                visualize_grasp(
                    vis, 'placement/gripper', placement,
                    [0, 255, 0], linewidth=cfg.meshcat.line_width
                )
                obj_placed = obj_pts @ placement[:3, :3].T + placement[:3, 3]
                visualize_pointcloud(
                    vis, 'placement/object', obj_placed,
                    [255, 0, 0], size=cfg.meshcat.point_size
                )
                input(f'Confidence {confidence[idx].item()}')
                if i >= cfg.eval.num_samples - 1:
                    break
        else:
            print('No placement')
        plt.close(fig)
        return region
def center_generator(poses, contacts):
    center = (contacts.min(axis=0) + contacts.max(axis=0)) / 2
    order = np.argsort(np.linalg.norm(contacts - center, axis=1))
    for i in order:
        yield poses[i], i

def fps_generator(poses, first):
    ctr_pts = load_control_points().unsqueeze(0)
    yield poses[first], first
    pts_selected = poses[first] @ ctr_pts
    selected = torch.zeros(poses.shape[0]).bool()
    selected[first] = True
    while not selected.all():
        pts_unselected = poses[~selected] @ ctr_pts
        diff = pts_selected.unsqueeze(0) - pts_unselected.unsqueeze(1)
        dist = diff[:, :, :3, :].norm(dim=2).sum(dim=2).min(dim=1)[0]
        next = torch.where(~selected)[0][dist.argmax()]
        yield poses[next], next
        selected[next] = True
        pts_selected = torch.cat([
            pts_selected, pts_unselected[next:next+1]
        ])


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    obs_cfg = create_obs_config(
        cfg.rlbench.cameras, cfg.rlbench.camera_resolution
    )
    demo = get_stored_demos(
        amount=1, image_paths=False, dataset_root=cfg.rlbench.demo_path,
        variation_number=-1, task_name=cfg.rlbench.task_name,
        obs_config=obs_cfg, random_selection=False,
        from_episode_number=cfg.rlbench.episode_id
    )[0]
    frame_id = cfg.rlbench.frame_id
    if frame_id is None:
        frame_id = np.random.randint(len(demo))
    demo = vars(demo[frame_id])

    model = M2T2(**M2T2.from_config(cfg.m2t2))
    ckpt = torch.load(cfg.eval.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()
    print("Model loaded from", cfg.eval.checkpoint)

    fig = plt.figure()
    plt.imshow(demo[f'{cfg.eval.ui_camera}_rgb'])
    fig.canvas.mpl_connect(
        'button_press_event', partial(select_object, cfg, demo, model, fig)
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
