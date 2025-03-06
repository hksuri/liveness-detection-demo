"""
data_preprocessing.py

Script to prepare and preprocess data from the Replay-Attack dataset 
(or your chosen dataset).
"""

import os
import cv2
import glob
import numpy as np

def extract_frames_from_video(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at a specified frame_rate (frames per second).
    Saves frames in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save a frame only at the specified rate
        if frame_count % int(fps // frame_rate) == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1
        frame_count += 1
    
    cap.release()

def prepare_dataset(dataset_root, output_root, frame_rate=1):
    """
    Example function to go through the dataset root (with real/spoof videos)
    and extract frames into a structure ready for training.
    """
    # Assumed directory structure for dataset
    # dataset_root/
    #    real/
    #        video1.mp4
    #        ...
    #    spoof/
    #        videoX.mp4
    #        ...
    
    for label in ["real", "spoof"]:
        video_paths = glob.glob(os.path.join(dataset_root, label, "*.mp4"))
        for video_path in video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(output_root, label, video_name)
            extract_frames_from_video(video_path, output_dir, frame_rate=frame_rate)

if __name__ == "__main__":
    # Example usage
    dataset_root = "path/to/Replay-Attack"
    output_root  = "path/to/preprocessed_data"
    prepare_dataset(dataset_root, output_root, frame_rate=1)