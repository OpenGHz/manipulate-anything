import argparse
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    JointPosition,
    EndEffectorPoseViaPlanning,
)
from rlbench.action_modes.gripper_action_modes import Discrete, GripperJointPosition
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.tasks import *  # noqa: F403


parser = argparse.ArgumentParser()
parser.add_argument("-tn", "--task-name", type=str, default="ReachTarget")
parser.add_argument(
    "--seed", type=int, default=None, help="Random seed for reproducibility"
)
args = parser.parse_args()

np.random.seed(args.seed)


class Agent(object):
    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()
    ),
    obs_config=ObservationConfig(),
    headless=True,
    dataset_root=""
)
env.launch()

task = env.get_task(locals()[args.task_name])

agent = Agent(env.action_shape)

rounds = 2
steps = 40
try:
    for r in range(rounds):
        input(f"Round {r + 1}/{rounds}. Press Enter to continue...")
        obs: Observation
        descriptions, obs = task.reset()
        for s in range(steps):
            # print(f"Step {s + 1}/{steps}")
            print(obs.left_shoulder_rgb.shape)
            action = agent.act(obs)
            print(action)
            obs, reward, terminate = task.step(action)
except KeyboardInterrupt:
    print("Interrupted by user")

print("Done")
env.shutdown()
