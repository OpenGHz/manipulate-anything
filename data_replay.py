import argparse
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from pyrep.const import RenderMode
from types import SimpleNamespace
from rlbench.backend import utils
import os


def get_observation_config(rlbench_cfg):
    img_size = rlbench_cfg.camera_resolution
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if rlbench_cfg.renderer == "opengl":
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL
    return obs_config


parser = argparse.ArgumentParser()
parser.add_argument("-tn", "--task-name", type=str, default="close_box")
parser.add_argument(
    "--seed", type=int, default=None, help="Random seed for reproducibility"
)
parser.add_argument(
    "-d",
    "--dataset-root",
    type=str,
    default="data/GT_12T_data/train_12_GT",
    help="Path to dataset root",
)
parser.add_argument("--live-demos", action="store_true", help="Use live demos")
parser.add_argument("--num-demos", type=int, default=2, help="Number of demos to fetch")
args = parser.parse_args()

np.random.seed(args.seed)

data_root = os.path.abspath(args.dataset_root)
task_name = args.task_name
live_demos = args.live_demos

if not live_demos:
    assert os.path.exists(data_root), f"Dataset root {data_root} does not exist."
# data_path = os.path.join(data_root, task_name)

obs_config = get_observation_config(
    SimpleNamespace(camera_resolution=(256, 256), renderer=None)
)

env = Environment(
    action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
    dataset_root=data_root,
    obs_config=obs_config,
    headless=False,
)
env.launch()

task_env = env.get_task(utils.task_file_to_task_class(task_name))
possible_variations = task_env.variation_count()
task_env.set_variation(-1)

np.set_printoptions(suppress=True)

demos = task_env.get_demos(-1, live_demos=live_demos)  # -> List[List[Observation]]
print(f"Fetched {len(demos)} demos.")
first_demo = demos[0]
print(first_demo)
first_obs = first_demo[0]
print(f"First obs info: {first_obs.left_shoulder_rgb.shape}")

rounds = len(demos)
try:
    for r in range(rounds):
        demo = demos[r]
        steps = len(demo)
        if input(
            f"Round {r + 1}/{rounds}. Max steps: {steps}. Press Enter to continue..."
        ):
            break
        variation = np.random.randint(possible_variations)
        # task_env.set_variation(variation)
        descriptions, obs = task_env.reset()
        obs: Observation
        for s in range(steps):
            point: Observation = demo[s]
            print(f"Step {s + 1}/{steps}")
            print(f"{point.gripper_pose=}, {point.gripper_open=}")
            action = np.concatenate(
                [point.gripper_pose, [point.gripper_open, 1.0]], axis=0
            )
            assert action.shape == (9,), f"Action shape mismatch: {action.shape}"
            obs, reward, terminate = task_env.step(action)
            success, terminate = task_env._task.success()
            if success:
                print("Task succeeded!")
                break
        else:
            print("Task did not succeed within the step limit.")
except KeyboardInterrupt:
    print("Interrupted by user")

print("Done")
env.shutdown()
