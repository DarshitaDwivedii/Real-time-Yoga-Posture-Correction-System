import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

DATASET_PATH = "Videos"
POSE_FOLDERS = os.listdir(DATASET_PATH)
landmarks_data = []

print("Starting data preprocessing with VIEW SEGMENTATION...")

for pose_name in POSE_FOLDERS:
    pose_path = os.path.join(DATASET_PATH, pose_name)
    if not os.path.isdir(pose_path): continue

    # --- THIS IS THE NEW, CORRECT LOGIC ---
    # Loop through all subfolders like 'Right_Front', 'Wrong_Side', etc.
    for subfolder_name in os.listdir(pose_path):
        
        # Determine the label parts from the folder name by splitting it
        parts = subfolder_name.split('_')
        # We expect 2 parts after the main pose name, e.g., ['Right', 'Front']
        # This check ensures we only process correctly named folders
        if len(parts) != 2: 
            print(f"Skipping folder with unexpected name: {subfolder_name}")
            continue
        
        correctness_label = parts[0]  # e.g., 'Right' or 'Wrong'
        view_label = parts[1]         # e.g., 'Front', 'Back', 'Side'
        
        # Create the final, detailed label
        final_label = f"{pose_name}_{correctness_label}_{view_label}"
        
        videos_folder_path = os.path.join(pose_path, subfolder_name)
        if not os.path.isdir(videos_folder_path): continue
        
        video_files = [f for f in os.listdir(videos_folder_path) if f.endswith(('.mp4', '.mov', '.avi'))]
        print(f"Processing {len(video_files)} videos for label: {final_label}")
        
        for video_file in video_files:
            video_path = os.path.join(videos_folder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video file {video_path}")
                continue
                
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)
                    
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        row = [lm.x for lm in landmarks] + [lm.y for lm in landmarks] + [lm.z for lm in landmarks] + [lm.visibility for lm in landmarks]
                        landmarks_data.append([final_label] + row)
                except Exception as e:
                    print(f"Error on frame in {video_path}: {e}")
            cap.release()

# --- The rest of the script to save the CSV is the same ---
print("Preprocessing finished.")
if landmarks_data:
    columns = ["label"]
    for i in range(33): columns.append(f'x{i}')
    for i in range(33): columns.append(f'y{i}')
    for i in range(33): columns.append(f'z{i}')
    for i in range(33): columns.append(f'vis{i}')

    df = pd.DataFrame(landmarks_data, columns=columns)
    OUTPUT_CSV_NAME = "yoga_landmarks_segmented.csv" # Using a new name
    df.to_csv(OUTPUT_CSV_NAME, index=False)
    print(f"Landmarks from {len(landmarks_data)} frames saved to {OUTPUT_CSV_NAME}")
else:
    print("CRITICAL: No landmark data was collected.")

pose.close()