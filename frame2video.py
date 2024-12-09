import cv2
import os
from natsort import natsorted
name='put_objects_in_container'
def images_to_video(folder_path, output_path, fps=30):
    # Get all files from the folder
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort files naturally
    files = natsorted(files)

    # Check if we have any files to process
    if not files:
        print("No images found in the specified directory!")
        return

    # Find out the frame size from the first image
    frame = cv2.imread(files[0])
    h, w, layers = frame.shape
    size = (w, h)

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    # Read each file and write it to the video
    for file in files:
        img = cv2.imread(file)
        out.write(img)

    # Release the VideoWriter
    out.release()
    print(f"Video saved to {output_path}")


for i in range(1,25):
    folder = '/home/duanj1/m2t2/manipulate_anything/RLBench/val/rlbench_new/'+str(name)+'/all_variations/episodes/episode'+str(i)+'/front_rgb'
    folder2 = '/home/duanj1/m2t2/manipulate_anything/RLBench/val/rlbench_new/'+str(name)+'/all_variations/episodes/episode'+str(i)+'f/front_rgb'
    try:
        # Try to open the file at the primary path
        output = '/home/duanj1/m2t2/manipulate_anything/RLBench/val/rlbench_new/'+str(name)+'/video/'+str(i)+'.mp4'
        images_to_video(folder, output)
        
    except FileNotFoundError:
        # If the file is not found at the primary path, try the secondary path
        try:
            output = '/home/duanj1/m2t2/manipulate_anything/RLBench/val/rlbench_new/'+str(name)+'/video/'+str(i)+'.mp4'
            images_to_video(folder2, output)
        except FileNotFoundError:
            print("File not found at both primary and secondary paths.")
        except Exception as e:
            print(f"An error occurred while trying to read the file from the secondary path: {e}")
    except Exception as e:
        print(f"An error occurred while trying to read the file from the primary path: {e}")
    
    
