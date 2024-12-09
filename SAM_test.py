import os
import cv2
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import supervision as sv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image
from translate import Translator
import cv2
import ast
import os
import glob
from transformers import pipeline
import subprocess
import os
import openai
# openai.organization = "org-98VX2pHuWcC8REfZpTt46ni0"
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

# print(openai.Model.list())
imagepath='/home/duanj1/m2t2/manipulate_anything/RLBench/saved_image.png'

import base64
import requests

# OpenAI API Key
api_key = key
# api_key = ""

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = imagepath
def GPTV4_Verification_Sinlge(prompt_text,folder_path):
    def find_latest_image(folder_path):
        # Search for image files (jpg, png, etc.) in the folder
        image_files = glob.glob(os.path.join(folder_path, '*.[pP][nN][gG]')) + \
                    glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) + \
                    glob.glob(os.path.join(folder_path, '*.[jJ][pP][eE][gG]'))

        # Sort the files by modification time in descending order
        image_files.sort(key=os.path.getmtime, reverse=True)

        # Return the most recent image if available
        return image_files[0] if image_files else None
    folder_path=folder_path
    print("HEREEEEE")
    print(folder_path)

    most_recent_image = folder_path
    print("Most Recent Image:", most_recent_image or "No image found.")

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
                    "content": [{"type": "text", "text": prompt_text}, image_message]
                }
            ],
            "max_tokens": 300
        }

        # Sending the request (assuming requests is imported and api_key is defined)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        output = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
        print('first GPT4:' + str(output))
        return str(output)
    else:
        return "No image found to process."

content2 ="""
"type": "text",
"text": f"There is a picture of robot manipulation scene.",
"text": f"Each frame are annotated with numbers at the different area of the image.",
"text": f"Select all the numbers out of all the numbers labelled in the image that are most relevant for reasoning and analysing whether the verification condition has been fullfilled.",
"text": f"Output the selected numbers into a list, and no other text in the output."
verification condition:
"""

file_path = '/home/duanj1/m2t2/manipulate_anything/RLBench/action_info.txt'
with open(file_path, 'r') as file:
    content = file.read()
    data_dict = ast.literal_eval(content)
next_verification=data_dict['verification'][0]
verification_prompt=str(content2)+str(next_verification) 


def increase_bbox(bbox, scale_factor=1):
    """
    Increase the bounding box size by a scale factor.

    :param bbox: A list of coordinates [x1, y1, x2, y2] for the bounding box
    :param scale_factor: Factor by which to increase the bounding box size
    :return: A list of coordinates [x1, y1, x2, y2] for the scaled bounding box
    """
    x1, y1, x2, y2 = bbox

    # Calculate the original width and height
    original_width = x2 - x1
    original_height = y2 - y1

    # Compute the center of the bounding box
    center_x = x1 + original_width / 2
    center_y = y1 + original_height / 2

    # Calculate the new width and height
    new_width = original_width * scale_factor
    new_height = original_height * scale_factor

    # Compute the new bounding box coordinates
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)

    return [new_x1, new_y1, new_x2, new_y2]
def draw_bounding_box(image_path, bounding_box, output_path='./output_image_verification.png'):
    """
    Draw a bounding box on an image.

    :param image_path: Path to the input image file
    :param bounding_box: A list of coordinates [x1, y1, x2, y2] for the bounding box
    :param output_path: Path for the output image file with the drawn bounding box
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"The specified image file was not found: {image_path}")

    # Define the bounding box coordinates
    x1, y1, x2, y2 = bounding_box

    # Draw the bounding box on the image
    # You can change the color and thickness of the bounding box here
    color = (0, 255, 0)  # Green color
    thickness = 3
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Save the output image with the bounding box
    cv2.imwrite(output_path, image)

    return output_path


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint='/home/duanj1/m2t2/sam_vit_h_4b8939.pth').to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
MIN_AREA_PERCENTAGE = 0.005  # 1% of the image area
MAX_AREA_PERCENTAGE = 0.4  # 50% of the image area


# load image
image_bgr = cv2.imread('/home/duanj1/m2t2/manipulate_anything/RLBench/saved_image.png')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# segment image
sam_result = mask_generator.generate(image_rgb)
detections = sv.Detections.from_sam(sam_result=sam_result)

# filter masks
height, width, channels = image_bgr.shape
image_area = height * width

min_area_mask = (detections.area / image_area) > MIN_AREA_PERCENTAGE
max_area_mask = (detections.area / image_area) < MAX_AREA_PERCENTAGE
detections = detections[min_area_mask & max_area_mask]

# setup annotators
mask_annotator = sv.MaskAnnotator(
    color_lookup=sv.ColorLookup.INDEX,
    opacity=0.0
)

# Adjust these parameters to change the size and thickness of the label text
text_scale = 0.5  # Smaller value makes the text (and box) smaller
text_thickness = 1  # Smaller value makes the text thinner

label_annotator = sv.LabelAnnotator(
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER,
    text_scale=text_scale,  # Use the adjusted text scale here
    text_color=sv.Color.WHITE,
    text_padding= 2,
    color=sv.Color.BLACK,  
    text_thickness=text_thickness  # Use the adjusted text thickness here
)

# annotate
labels = [str(i) for i in range(len(detections))]
print(labels)
bounding_boxes = []  # List to store bounding box coordinates

for i, det in enumerate(detections):
    bbox = det[0]  # Extract the bounding box from the first element of the tuple
    x1, y1, x2, y2 = bbox  # Unpacking the bounding box coordinates

    # Append the bounding box coordinates as a sublist to the bounding_boxes list
    bounding_boxes.append([x1, y1, x2, y2])

    # print(f"Label {i}: Top-left (x={x1}, y={y1}), Bottom-right (x={x2}, y={y2})")
# print(bounding_boxes[12])


# print(f"Output image saved at: {output_image_path}")

annotated_image = mask_annotator.annotate(
    scene=image_bgr.copy(), detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

save_path = './annotated_image.png'
cv2.imwrite(save_path, annotated_image)

response=GPTV4_Verification_Sinlge(verification_prompt,save_path)
print(type(list(response)))
with open(file_path, 'r') as file:
    content = file.read()
    data_dict = ast.literal_eval(content)
data_dict['SoM']=(response)

image_path = '/home/duanj1/m2t2/manipulate_anything/RLBench/saved_image.png'
from PIL import Image, ImageDraw, ImageFont
import math
def label_image(image, label, position=(50, 50), radius=30, opacity=0.5):
    # Create a circle on a separate image (overlay)
    overlay = image.copy()
    cv2.circle(overlay, position, radius, (255, 255, 255), -1)  # -1 fills the circle

    # Alpha blend the overlay with the original image
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

    # Calculate the font scale and thickness
    font_scale = radius / 30  # Adjust font scale based on radius
    thickness = 2

    # Get the size of the text box
    text_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

    # Calculate the text position so it's centered
    text_x = position[0] - text_size[0] // 2
    text_y = position[1] + text_size[1] // 2

    # Put the text in the image
    cv2.putText(image, str(label), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

def draw_circle_with_label(image_path, coordinates, label='1', save_path='labeled_image.jpg'):
    # Calculate the center, width, and height of the rectangle
    x1, y1, x2, y2 = coordinates
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # Set the diameter of the circle to fit within the rectangle, but smaller
    diameter = min(width, height) * 0.5  # make the circle slightly smaller
    radius = int(diameter / 2)

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Label the image with the circle and number, adjusting opacity
    label_image(image, label, position=center, radius=radius, opacity=0.5)  # Adjust opacity as needed

    # Save the modified image to the specified path
    cv2.imwrite(save_path, image)

    return save_path

# Example usage
response_list = ast.literal_eval(response)
bounding_box = bounding_boxes[(response_list[0])]  # Example coordinates

# saved_image_path = draw_circle_with_label(image_path, bounding_box, label='1', save_path='/home/duanj1/m2t2/manipulate_anything/RLBench/output_image_verification.png')

bounding_box_new = increase_bbox(bounding_box, 3)

output_image_path = draw_bounding_box(image_path, bounding_box_new)

with open('/home/duanj1/m2t2/manipulate_anything/RLBench/action_info.txt', 'w') as f:
            # f.write(str(empty))
    f.write(str(data_dict))
