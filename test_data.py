import pickle
from rlbench.backend.observation import Observation
from pathlib import Path

tasks_dir = Path("data/GT_12T_data/train_12_GT")
print(f"Tasks: {list(tasks_dir.iterdir())}")

for task in tasks_dir.iterdir():
    # print(task.name)
    episodes_dir = (
        "data/GT_12T_data/train_12_GT/{task_name}/all_variations/episodes/".format(
            task_name=task.name
        )
    )
    for episode in Path(episodes_dir).iterdir():
        # print(episode.name)
        data_path = "{episode}/low_dim_obs.pkl".format(episode=episode)
        with open(data_path, "rb") as f:
            low_dim_obs: list[Observation] = pickle.load(f)
            low_dim_length = len(low_dim_obs)
            print(low_dim_length)
            print(low_dim_obs[0].gripper_pose)
            # print(low_dim_obs[0].joint_positions)
            # print(low_dim_obs[0].gripper_pose)
        for image_folder in episode.iterdir():
            # print(image_folder.name)
            if image_folder.is_dir():
                assert len(list(image_folder.iterdir())) == low_dim_length, (
                    f"Image folder {image_folder} does not match low_dim_obs length {low_dim_length}"
                )
    #     break
    # break