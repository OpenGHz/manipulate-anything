from m2t2.m2t2 import M2T2
from m2t2.meshcat_utils import (
    create_visualizer, make_frame,
    visualize_grasp, visualize_pointcloud
)
from m2t2.dataset_utils import sample_points
from m2t2.model_utils import to_cpu, to_gpu
from m2t2.rlbench_utils import (
    pcd_rgb_within_bound, gripper_pose_from_rlbench,
    gripper_pose_to_rlbench, rotation_to_rlbench
)
from scipy.spatial import KDTree
from yarr.agents.agent import Agent
import numpy as np
import torch
import random
import matplotlib.cm as cm
import ast
from skill_library import (  press_action, rotate_gripper_y, 
    sweep_action, rotate_gripper_z, rotate_gripper_x, 
    drop_action, close_gripper_action, open_gripper_action
)


class M2T2Agent(Agent):
    def __init__(self, cfg):
        super(M2T2Agent, self).__init__()
        self.cfg = cfg
        self.prev_state = 'None'
        self.prev_prev_state = 'None'
        self.action_history = ['None']
        self.model = M2T2(**M2T2.from_config(self.cfg.m2t2))
        self.vis = None
        self.pick_called =False
        if self.cfg.meshcat.visualize:
            self.vis = create_visualizer()

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return [w, x, y, z]
    
    def set_pick_called(self, value):
        self.pick_called = value

    def run_code_from_file(self,file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # Remove the first and last lines
                code = ''.join(lines[1:-1])

                # print("CODE TO RUN:")
                # print(code)
                exec(code)
        except FileNotFoundError:
            print(f"The file {file_path} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


    def load_weights(self):
        ckpt = torch.load(self.cfg.eval.checkpoint, map_location='cpu')
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.cuda().eval()

    

    def act(self, obs, state):
        action = obs['gripper_matrix']
        gripper_open = obs.get('gripper_open', 0)
        trans = action[:3, 3]  # Initial translation based on gripper matrix
        rot = obs['gripper_pose'][3:]  # Initial rotation based on gripper pose
        pred = None  # Initialize pred to avoid issues
        
        
        


        pcd_raw, rgb_raw, seg_raw = pcd_rgb_within_bound(
            obs, self.cfg.rlbench.cameras, self.cfg.rlbench.scene_bounds
        )
        if self.vis is not None:
            self.vis.delete()
            visualize_pointcloud(
                self.vis, 'scene', pcd_raw, rgb_raw,
                size=self.cfg.meshcat.point_size
            )
            # for i in np.unique(seg_raw):
            #     visualize_pointcloud(
            #         self.vis, f'seg_{i}', pcd_raw[seg_raw == i],
            #         rgb_raw[seg_raw == i], size=self.cfg.meshcat.point_size
            #     )
        self.init_state = state
        file_path = './action_info.txt'
        with open(file_path, 'r') as file:
            content = file.read()
            data_dict = ast.literal_eval(content)
        next_action = data_dict['primitive_actions'][0]
        with open('./action_code.txt', 'r') as file:
            lines = file.readlines()

        # Remove the first and last lines
        lines = lines[1:-1]

        # Join the remaining lines back into a single string
        function_code = ''.join(lines)

        # Execute the function code so that the function is available in the current script
        exec(function_code)


        
        if state == str(next_action) and str(next_action) != 'pick' and str(next_action) != 'place':
            
            function_name= str(next_action)
            if function_name in locals():
                action, gripper_open,pred = locals()[function_name](obs, self.cfg)
        
        elif state == 'press':
            action, gripper_open,pred = press_action(obs, self.cfg)


        elif state == 'rotatey':
            action, gripper_open,pred = rotate_gripper_y(obs, self.cfg)
        
        elif state == 'sweep':
            action, gripper_open,pred = sweep_action(obs, self.cfg)

        elif state == 'rotatez':
            action, gripper_open,pred = rotate_gripper_z(obs, self.cfg)

        elif state == 'rotatex':
           action, gripper_open,pred = rotate_gripper_x(obs, self.cfg)

        elif state == 'drop':
            action, gripper_open,pred = drop_action(obs, self.cfg) 


        elif state == 'success':
            action = obs['gripper_matrix']
            pred = np.array([
                [-0.970757, 0, 0.240063, 0.278442],
                [0, 1.0, 0, -0.008160],
                [-0.240063, 0, -0.970757, 1.472001],
                [0, 0, 0, 1]
            ])
            gripper_open = 1
            message = 'Double retract detected. Reset.'
            trans = action[:3, 3]
            rot = obs['gripper_pose'][3:]

    
        elif state == 'close':
            action, gripper_open,pred = close_gripper_action(obs, self.cfg) 

        elif state == 'open':
            action, gripper_open,pred = open_gripper_action(obs, self.cfg) 

           
        else:
            
            if state == 'pick':
                is_pick = torch.ones(1)
                is_place = torch.zeros(1)
                gripper_open = 0
                obj_pts = np.random.rand(self.cfg.data.num_object_points, 3) - 0.5
      
                    
            elif state == 'place':
                is_pick = torch.zeros(1)
                is_place = torch.ones(1)
                # gripper_open = 1
                from matplotlib import pyplot as plt
                # plt.imshow(obs['wrist_rgb'])
                # plt.tight_layout()
                # plt.show()
                bottom = obs['wrist_rgb'].shape[0] - 1
                center = obs['wrist_rgb'].shape[1] // 2
                obj_id = obs['wrist_mask'][bottom][center]
                if np.issubdtype(obj_id.dtype, np.floating):
                    obj_id = int(obj_id[0] * 255)
                obj_pts = pcd_raw[seg_raw == obj_id]
                obj_pt_idx = sample_points(obj_pts, self.cfg.data.num_object_points)
                obj_pts = obj_pts[obj_pt_idx]

            pt_idx = sample_points(pcd_raw, self.cfg.data.num_points)
            pcd, rgb, seg = pcd_raw[pt_idx], rgb_raw[pt_idx], seg_raw[pt_idx]
            ee_pose = gripper_pose_from_rlbench(obs['gripper_matrix'])
            inv_ee_pose = np.linalg.inv(ee_pose)
            obj_pts_ee = obj_pts @ inv_ee_pose[:3, :3].T + inv_ee_pose[:3, 3]
            if self.vis is not None and state == 'place':
                visualize_pointcloud(
                    self.vis, 'object', obj_pts, [255, 0, 0],
                    size=self.cfg.meshcat.point_size
                )
            inputs = {
                'inputs': torch.from_numpy(
                    pcd - pcd_raw.mean(axis=0)
                ).float().unsqueeze(0),
                'points': torch.from_numpy(pcd).float().unsqueeze(0),
                'task_is_pick': is_pick,
                'task_is_place': is_place,
                'object_inputs': torch.from_numpy(
                    obj_pts_ee - obj_pts_ee.mean(axis=0)
                ).float().unsqueeze(0),
                'object_points': [torch.from_numpy(obj_pts).float()],
                'ee_pose': torch.from_numpy(ee_pose).float().unsqueeze(0)
            }
            to_gpu(inputs)
            with torch.no_grad():
                outputs = self.model.infer(inputs, self.cfg.eval)
            to_cpu(outputs)
            if state == 'place':
                minx, miny = obs['target_box'][0]
                maxx, maxy = obs['target_box'][1]
                pts = obs[f'{self.cfg.eval.ui_camera}_point_cloud']
                pts = pts[miny:maxy, minx:maxx].reshape(-1, 3)
                
            else:
                pts = []
                for key, val in obs['target_box'].items():
                    # print("DEBUG")
                    # print(val)
                    # print(key)
                    minx, miny, maxx, maxy = val
                    pts.append(obs[f'{key}_point_cloud'][
                        miny:maxy, minx:maxx
                    ].reshape(-1, 3))
                pts = np.concatenate(pts, axis=0)
            
            bounds = self.cfg.rlbench.scene_bounds
            within = (pts[:, 0] > bounds[0]) & (pts[:, 0] < bounds[3]) \
                   & (pts[:, 1] > bounds[1]) & (pts[:, 1] < bounds[4]) \
                   & (pts[:, 2] > bounds[2]) & (pts[:, 2] < bounds[5])
            target = KDTree(pts[within])
            visualize_pointcloud(
                self.vis, 'target_box', pts[within],
                [0, 0, 255], size=self.cfg.meshcat.point_size
            )
            if state == 'pick':
                
                contacts = torch.cat(outputs['contacts'][0])
                grasps = torch.cat(outputs['grasps'][0])
                confidence = torch.cat(outputs['confidence'][0])
                dist, _ = target.query(contacts.numpy())
                matched = np.where(dist < self.cfg.eval.radius)[0]
                if len(matched) == 0:
                    pred = gripper_pose_from_rlbench(obs['gripper_matrix'])
                    message = 'No grasp detected'
                    gripper_open = obs['gripper_open']
                else:
                    conf, idx = confidence[matched].max(dim=0)
                    pred = grasps[matched][idx].numpy()
                    message = f'Grasp confidence {conf:.8f}'
                    gripper_open = 0
                
            elif state == 'place':
                visualize_pointcloud(
                    self.vis, 'placements', outputs['place_contacts'][0].numpy(),
                    [0, 255, 0], size=self.cfg.meshcat.point_size
                )
                dist, _ = target.query(outputs['place_contacts'][0])
                matched = np.where(dist < self.cfg.data.contact_radius)[0]
                if len(matched) == 0:
                    pred = gripper_pose_from_rlbench(obs['gripper_matrix'])
                    message = 'No placement inside region'
                    gripper_open = obs['gripper_open']

                else:
                    confidence = torch.cat([
                        outputs['place_confidence'][0][i] for i in matched
                    ])
                    placements = torch.cat([
                        outputs['placements'][0][i] for i in matched
                    ])

                    sorted_confidences, sorted_indices = torch.sort(confidence, descending=True)

                    # Check if there are at least two predictions to select the second most confident
                    if len(sorted_confidences) >= 2:
                        # Select the second most confident prediction
                        second_highest_confidence = sorted_confidences[1]
                        second_highest_index = sorted_indices[1]
                        pred = placements[second_highest_index].numpy()
                        message = f'Placement confidence {second_highest_confidence:.8f}'
                    else:
                        # Fallback to the highest confidence if there's only one prediction
                        highest_confidence = sorted_confidences[0]
                        highest_index = sorted_indices[0]
                        pred = placements[highest_index].numpy()
                        message = f'Placement confidence {highest_confidence:.8f}'

                    # gripper_open = 1
                    gripper_open = obs['gripper_open']
                    pred[:3, 3] = pred[:3, 3] - 0.005 * pred[:3, 2]



                    # conf, idx = confidence.max(dim=0)
                    # pred = placements[idx].numpy()
                    # print("Place:" +str(placements))
                    # message = f'Placement confidence {conf:.8f}'
                    # gripper_open = 1
                    # pred[:3, 3] = pred[:3, 3] - 0.02 * pred[:3, 2]
                    
            
            # action[:3, 3] = action[:3, 3] - self.cfg.eval.retract * action[:3, 2]
            # gripper_open = obs['gripper_open']
            # message = 'Retract'
            # pred = gripper_pose_from_rlbench(action)
            # trans = action[:3, 3]
            # rot = obs['gripper_pose'][3:]

            pose = gripper_pose_to_rlbench(pred)
            trans = pose[:3, 3]
            rot = rotation_to_rlbench(pose)
            #Non-Prehensile
            # pose = gripper_pose_to_rlbench(pred)
            # trans = pose[:3, 3]- 0.03 * pose[:3, 2]
            # rot = rotation_to_rlbench(pose)

        if self.vis is not None:
            visualize_grasp(
                self.vis, 'grasp', pred, color=[0, 255, 0],
                linewidth=self.cfg.meshcat.line_width
            )
            # input(message)
        ignore_collision = 0
        action = np.concatenate([trans, rot, [gripper_open, ignore_collision]])
       
        self.prev_prev_state = self.prev_state
        self.prev_state = state
        
        return action

    def build(self, training, device):
        pass

    def update(self):
        pass

    def act_summaries(self):
        pass

    def update_summaries(self):
        pass

    def save_weights(self, savedir):
        pass