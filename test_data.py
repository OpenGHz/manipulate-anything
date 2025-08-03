import pickle
from rlbench.backend.observation import Observation


data_path = "/home/ghz/Work/Research/manipulate-anything/data/GT_12T_data/train_12_GT/close_box/all_variations/episodes/episode2/low_dim_obs.pkl"


with open(data_path, "rb") as f:
    low_dim_obs: list[Observation] = pickle.load(f)

print(len(low_dim_obs))
print(low_dim_obs[0])
print(low_dim_obs[0].joint_positions)
print(low_dim_obs[0].gripper_pose)