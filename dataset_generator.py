from PIL import Image
from functools import partial
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from pyrep.const import RenderMode
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.demo import Demo
from rlbench.environment import Environment
import hydra
import numpy as np
import os
import pickle
import ast
import os
import shutil
from rlbench.backend import task as rlbench_task
from matplotlib.patches import Circle, Rectangle
import openai
import base64
import requests
import subprocess
from m2t2_agent import M2T2Agent
from MA_utils import check_and_make, read_values_from_txt, get_bbox, get_bbox2, encode_image
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key



# def check_and_make(dir):
#     if not os.path.exists(dir):
#         os.makedirs(dir)


def save_demo(demo, example_path, variation):
    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
        example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(
        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(
        example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(
        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(
        example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(
        example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(
        example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(
        example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(left_shoulder_depth_path)
    check_and_make(left_shoulder_mask_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(right_shoulder_depth_path)
    check_and_make(right_shoulder_mask_path)
    check_and_make(overhead_rgb_path)
    check_and_make(overhead_depth_path)
    check_and_make(overhead_mask_path)
    check_and_make(wrist_rgb_path)
    check_and_make(wrist_depth_path)
    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(
            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray(
            (obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(
            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray(
            (obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(
            obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray(
            (obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(
            obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(
            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(
            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(
            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(
            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(
            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        overhead_rgb.save(
            os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(
            os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(
            os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)


def check_state(task, variation, obs):
    file_name = './action_info.txt'
    values = read_values_from_txt(file_name)
    gripper_open = obs['gripper_open']

    with open(file_name, 'r') as f:
        data = f.read()
        data_dict = ast.literal_eval(data)  # Convert string representation of dictionary to actual dictionary
        primtivate_action =data_dict['primitive_actions'][0]
    print("ACTION:"+ str(primtivate_action))
    with open('./log.txt', 'a') as file:
            file.write("Action: "+str(primtivate_action) + "\n")
    if values == 1:
        state = str(primtivate_action)
    elif values == 2:
        state = str(primtivate_action)
    elif values == 3:
        state = str(primtivate_action)
    elif values == 4:
        state = str(primtivate_action)
    elif values == 5:
        state = str(primtivate_action)
    elif values == 6:
        state = str(primtivate_action)
    elif values == 7:
        state = str(primtivate_action)
    elif values == 8:
        state = str(primtivate_action)
    elif values == 9:
        state = str(primtivate_action)
    elif values == 10:
        state = str(primtivate_action) 
    elif values == 11:
        state = str(primtivate_action)
    elif values == 12:
        state = str(primtivate_action)  
    else:
        state = 'success' 
    print('gripper open', gripper_open)
    return state





def GPT_4V_Viewpoint():
     # folder_path = '/home/duanj1/m2t2/manipulate_anything/RLBench/combined_image.png'
    file_path = './action_info.txt'
    with open(file_path, 'r') as file:
        content = file.read()
        data_dict = ast.literal_eval(content)
    next_action = data_dict['primitive_actions'][0]
    next_obj =  data_dict['Objects'][0]

    most_recent_image = './combined_image_all.png'
    prompt = f"""
    "type": "text",
    "text": "There is a picture containing 4 frames of a robot manipulation scene at the same time step but observed from four different camera viewpoints.",
    "text": "Each frame is annotated with a number on the top left corner of the image, numbered from 0 to 3.",
    "text": "The frame with number 0 annotated refers to the front viewpoint, number 1 refers to the wrist viewpoint, number 2 refers to the left shoulder viewpoint, and number 3 refers to the right shoulder viewpoint.",
    "text": "Given that the robot agent is currently performing the sub-task: {next_action} {next_obj}, compare the four frames and select the one viewpoint that offers the least obstructed view  and could see the robot arm performing {next_action} {next_obj}.",
    "text": "Output should only be one number between 0 and 3 representing the different viewpoints, and written in a list format, for example .",
    "text": "For example, if the front viewpoint (0) provides the clearest view, the output should be: [0]."
    """

    print("MT-VIewpoint Selection Prompt (Object-Centric Action)....")
    if most_recent_image:
        # Encoding the image
        encoded_image = encode_image(most_recent_image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Preparing the message part of the payload with the single image
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, image_message]
                }
            ],
            "max_tokens": 300
        }

        # Sending the request (assuming requests is imported and api_key is defined)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # print(response.json())
        output = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        return str(output)
    else:
        return "No image found to process."


class Recorder(object):
    def __init__(self) -> None:
        self.observations = []

    def save(self, obs):
        self.observations.append(obs)

    def clear(self):
        self.observations = []


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    img_size = cfg.rlbench.camera_resolution
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

    if cfg.rlbench.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL
    

    recorder = Recorder()
    arm_action_mode = EndEffectorPoseViaPlanning()
    arm_action_mode.set_callable_each_step(recorder.save)
    rlbench_env = Environment(
        # action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        action_mode=MoveArmThenGripper(arm_action_mode, Discrete()),
        obs_config=obs_config,
        headless=True)
    rlbench_env.launch()

    task_env = rlbench_env.get_task(
        utils.task_file_to_task_class(cfg.rlbench.task_name)
    )
    possible_variations = task_env.variation_count()

    variation_path = os.path.join(
        cfg.rlbench.demo_path, task_env.get_name(), VARIATIONS_ALL_FOLDER
    )
    check_and_make(variation_path)

    episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
    check_and_make(episodes_path)

    np.set_printoptions(suppress=True)
    agent = M2T2Agent(cfg)
    agent.load_weights()
    # for ex_idx in range(cfg.rlbench.episodes_per_task):
    #Number of Eps
    def clear_file(file_path):
        try:
            # Open the file in write mode, which clears existing content
            with open(file_path, 'w') as file:
                file.write('')
            print(f"Content cleared successfully from '{file_path}'")
        except Exception as e:
            print(f"An error occurred: {e}")
    file_path_result ='./generated_data_output/result.txt'
    # clear_file(file_path_result)
    def clear_folder(folder_path):
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            else:
                print(f"The directory {folder_path} does not exist.")
        
    # clear_folder(Save_fold)
    file_path_log = './log.txt'  # Specify your file path here

    # Reading from the file
    Num_of_generated_ep = 1
    Max_steps_attempt=10
    Max_retries = 7
    print("Number of generated Ep: "+str(Num_of_generated_ep))
    print("Max action steps: "+str(Max_steps_attempt))
    print("Max action retries: "+str(Max_retries))

    for ex_idx in range(0,Num_of_generated_ep):
        

        
        def clear_folder(folder_path):
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            else:
                print(f"The directory {folder_path} does not exist.")
        


        folder_path_right = "./save_frames_right"
        folder_path_left = "./save_frames_left"
        folder_path_overhead = "./save_frames_overhead"
        folder_path_wrist = "./save_frames_wrist"

        # Clear both folders using the function
        clear_folder(folder_path_right)
        clear_folder(folder_path_left)
        clear_folder(folder_path_overhead)
        clear_folder(folder_path_wrist)

        folder_path ="./save_frames"
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        variation = np.random.randint(possible_variations)
        print("variations: " +str(variation))
        # variation = 0
        task_env.set_variation(variation)
        descriptions, obs = task_env.reset()
        observations = [obs]
        print(
            'Task:', task_env.get_name(), '// Variation:', variation,
            '// Demo:', ex_idx
        )
        counter = 0
        action_counter =0

        # with open('/home/duanj1/m2t2/manipulate_anything/RLBench/action_info.txt', 'r') as f:
        #     data = f.read()
        #     data_dict = ast.literal_eval(data)  # Convert string representation of dictionary to actual dictionary

        
        state_occurrences = {}
        succ = 0
        action_list = []

        with open(file_path_log, 'a') as file:
            file.write("Ep: "+str(ex_idx) + "\n")
        

        for step in range(cfg.rlbench.episode_length):
            print("STEP:" + str(counter))
            if counter == 1:
                subprocess.run(["python", "/home/duanj1/CameraCalibration/LLMs/Qwen-VL/planner.py"])
    

            with open(file_path_log, 'a') as file:
                file.write("Step: "+str(step) + "\n")

            viewpoint_prediction = ast.literal_eval(GPT_4V_Viewpoint())
            view_grasp= int(viewpoint_prediction[0])
            # with open('/home/duanj1/m2t2/manipulate_anything/RLBench/action_info.txt', 'w') as file:
            #     data_dict['viewpoints'][0]=(viewpoint_prediction[0])
            #     file.write(str(data_dict))
                # print("HERERE:"+str(data_dict))
            # data_dict['viewpoints'][0]=(viewpoint_prediction[0])
            # data_dict['viewpoints'][1]=(viewpoint_prediction[1])
            
            #AUTO
              # COMMENT HERE for manual
            # ask_to_continue()
            

            if counter > Max_steps_attempt:
                agent.set_pick_called(False)
                break

            if not isinstance(obs, dict):
                obs = vars(obs)
            # if counter ==0:
            #     state='drop'
            plt.imshow(obs[f'{cfg.eval.ui_camera}_rgb'])
            plt.axis('off')
            plt.tight_layout(pad=0)
            save_path = "saved_image.png"
            saved_path2='./save_frames/'+str(step)+'.png'
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.savefig(saved_path2, bbox_inches='tight', pad_inches=0)

            plt.draw()
            plt.clf()  # Clear the figure for the next image
            def save_image(obs, key, step, base_path):
                plt.imshow(obs[key])
                plt.axis('off')
                plt.tight_layout(pad=0)
                saved_path = f'{base_path}/{step}.png'
                plt.savefig(saved_path, bbox_inches='tight', pad_inches=0)
                plt.draw()

            save_image(obs, 'left_shoulder_rgb', step, './save_frames_left')
            save_image(obs, 'right_shoulder_rgb', step, './save_frames_right')
            save_image(obs, 'overhead_rgb', step, './save_frames_overhead')
            save_image(obs, 'wrist_rgb', step, './save_frames_wrist')
            
            subprocess.run(["python", "./combined_all_image.py"])
            subprocess.run(["python", "/home/duanj1/CameraCalibration/LLMs/Qwen-VL/locate.py"])
            state = check_state(task_env._task, variation, obs)
            print("Current state: "+str(state))
            if state != 'pick' and state != 'place' and state != 'success':
                print("Generating action code.....")
                subprocess.run(["python", "/home/duanj1/CameraCalibration/LLMs/Qwen-VL/action_generation.py"])

            action_list.append(state)
            success, terminate = task_env._task.success()
            if success:
                print("Demo was a success!")
                succ = 1
                with open('./log.txt', 'a') as file:
                    file.write("Succeed!"+"\n")
                break

            else:
                print("Not succeed yet")
  
            # Update the state occurrences
            state_occurrences[state] = state_occurrences.get(state, 0) + 1

            # print()

            # Check if any state has occurred more than x times
            if state_occurrences[state] > Max_retries:
                print(f"State '{state}' called more than set number of times, breaking the loop.")
                break

            # Rest of your existing code...
            if state == 'success':
                agent.set_pick_called(False)
                break
            if state == 'pick':
                # obs['target_box'] = get_bbox(obs, cfg)
                obs['target_box'] = get_bbox(view_grasp)
                # print("FINAL Grasp: "+str(obs['target_box']))
            if state == 'place':
                obs['target_box'] = get_bbox2(state)
            # action = agent.act(obs, state)
            try:
                # Code that might cause an error
                # For example:
                print("COUNTER: "+str(counter))
                action = agent.act(obs, state)
                
                # if (action_list[-1] =='pick' and counter > 2) or action_list[-1] != 'place':
                #     pass
                # else:
                #     action = agent.act(obs, state)
                print(action_list[-1])
                action_list.append(state)
            except Exception as e:
                # This block will be executed if an error occurs
                print(f"An error occurred: {e}")
                break



            try:
                obs, reward, terminate = task_env.step(action)
                observations.append(obs)
                
            except Exception as e:
                print(e)
            counter = counter + 1
        empty={'primitive_actions': ['pick'], 'Objects': ['None'], 'pick': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 'place': [0, 0, 0, 0], 'predict': 1, 'verification': ['no'], 'viewpoints': [0, 1]}
        with open('./action_info.txt', 'w') as f:
            f.write(str(empty))
            # f.write(str(data))


        episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
        # save_demo(Demo(observations), episode_path, variation)

        save_demo(Demo(recorder.observations), episode_path, variation)
        with open(os.path.join(episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)
        file_path ='./generated_data_output/result.txt'
        with open(file_path, 'a') as file:  # 'a' mode for appending to the file
            file.write(f"Ep: {ex_idx}, Success: {succ}\n")


        
        recorder.clear()
    rlbench_env.shutdown()


if __name__ == '__main__':
    main()