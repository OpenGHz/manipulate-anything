import os
import cv2
import numpy as np

def find_most_recent_image_in_folder(folder_path):
    img_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        return None
    most_recent_file = max(img_files, key=lambda f: os.path.getctime(os.path.join(folder_path, f)))
    return os.path.join(folder_path, most_recent_file)

def label_image(image, label, position=(50, 50), radius=30):
    cv2.circle(image, position, radius, (255, 255, 255), -1)  # -1 fills the circle
    cv2.putText(image, str(label), (position[0] - 10, position[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

folders = [
    '/home/duanj1/m2t2/manipulate_anything/RLBench/save_frames',
    '/home/duanj1/m2t2/manipulate_anything/RLBench/save_frames_wrist',
    '/home/duanj1/m2t2/manipulate_anything/RLBench/save_frames_left',
    '/home/duanj1/m2t2/manipulate_anything/RLBench/save_frames_right'
]

# Find the most recent image in each folder and load them
images = []
for i, folder in enumerate(folders):
    image_path = find_most_recent_image_in_folder(folder)
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            label_image(image, i)
            images.append(image)

# Combine the images
if images:
    combined_image = np.hstack(images)
    output_path = './combined_image_all.png'
    cv2.imwrite(output_path, combined_image)
    
else:
    print("No images found or loaded.")
