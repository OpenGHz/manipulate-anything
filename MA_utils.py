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
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_values_from_txt(file_name):
    with open(file_name, 'r') as f:
        data = f.read()
        data_dict = ast.literal_eval(data)  # Convert string representation of dictionary to actual dictionary
        return data_dict['predict']

def ask_to_continue():
    response = input("Do you want to continue? (Type 'exit' to terminate): ").strip().lower()
    if response == 'exit':
        print("Terminating the program.")
        exit()



def parse_line(line):
    label, values_str = line.split('=')
    if 'predict' in label:
        return label, int(values_str.strip())
    values = values_str.strip('[]').split(', ')
    coords = [[int(values[0]), int(values[1])], [int(values[2]), int(values[3])]]
    return label, coords

def read_bboxes_from_txt(index,file_name):
    number_grasp=index
    list_randon=[]
    list_randon.append(number_grasp)
    # print("KKKKKKNNNNNN:"+str(list_randon))
    with open(file_name, 'r') as f:
        data = f.read()
        data_dict = ast.literal_eval(data)  # Convert string representation of dictionary to actual dictionary
        pick_coords = data_dict['pick']
        
        # Check if 'pick_coords' is a list of lists and assign to appropriate keys
        selected_k = list_randon
        print(selected_k)
        first_nu = selected_k[0]
        # second_nu = selected_k[1]

        # Assuming selected_keys is a list of keys you want to include in the result
        allview = ['front', 'wrist', 'left_shoulder', 'right_shoulder']  # This can be set externally
        selected_keys = []
        selected_keys.append(str(allview[first_nu]))
        # selected_keys.append('right_shoulder')

        with open('/home/duanj1/m2t2/manipulate_anything/RLBench/log.txt', 'a') as file:
            file.write("Viewpoint for grasp: "+str(allview[first_nu]) + "\n")
            

        # selected_keys.append(str(allview[second_nu]))
        print(selected_keys)
        if all(isinstance(item, list) for item in pick_coords):
            keys = ['front', 'wrist', 'left_shoulder', 'right_shoulder']
            # Filter keys based on selected_keys and create the result dictionary
            result = {key: pick_coords[keys.index(key)] for key in keys if key in selected_keys}

        else:
            result = {'front': pick_coords}  # Fallback to original behavior for a single list
        print("GRASP_View: "+str(result))
        return result

def get_bbox(number):
    file_name = '/home/duanj1/m2t2/manipulate_anything/RLBench/action_info.txt'
    index_grasp=number
    # print("NNNNNN:"+str(index_grasp))
    bboxes = read_bboxes_from_txt(index_grasp,file_name)
    # print("HERERERERER:")
    # print(bboxes)
    return bboxes

def read_bboxes_from_txt2(file_name):
    with open(file_name, 'r') as f:
        data = f.read()
        data_dict = ast.literal_eval(data)  # Convert string representation of dictionary to actual dictionary
        # pick_coords = data_dict['pick']
        place_coords = data_dict['place']

        # Transforming the list into the format you've mentioned
        # pick_transformed = [[pick_coords[i], pick_coords[i+1]] for i in range(0, len(pick_coords), 2)]
        # print(pick_transformed)
        place_transformed = [[place_coords[i], place_coords[i+1]] for i in range(0, len(place_coords), 2)]

        return {'place': place_transformed}


def get_bbox2(state):
    file_name = '/home/duanj1/m2t2/manipulate_anything/RLBench/action_info.txt'
    bboxes = read_bboxes_from_txt2(file_name)
    return bboxes[state]
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
