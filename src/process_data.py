import cv2
import mediapipe as mp
import os
import csv
import numpy as np
from pathlib import Path

def process_videos(data_path, output_csv_path):
    """
    Processes all videos from data_path, extracts key points for legs,
    normalizes them, and saves them to a CSV file, including the video file name.
    """
    # Initialize the MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Definition of key points for legs
    LOWER_BODY_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    ]

    # Create header for CSV file
    header = ['video_id', 'class']
    for landmark in LOWER_BODY_LANDMARKS:
        header.extend([f'{landmark.name}_x', f'{landmark.name}_y', f'{landmark.name}_z'])

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Loop through all folders (classes)
        for class_name in sorted(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            print(f"--- Processing class: {class_name} ---")

            # Support for subfolders
            for root, _, files in os.walk(class_path):
                for video_file in sorted(files):
                    if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                        continue
                    
                    video_path = os.path.join(root, video_file)
                    video_id = Path(video_path).stem  # We use the filename without the extension as the ID
                    print(f"  -> Processing video: {video_file}")
                    
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"     Error opening video file: {video_file}")
                        continue

                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        # Color conversion and frame processing
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False # Improves performance
                        results = pose.process(image)
                        image.flags.writeable = True

                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            
                            all_landmarks_visible = all(landmarks[lm.value].visibility > 0.6 for lm in LOWER_BODY_LANDMARKS)
                            
                            if all_landmarks_visible:
                                # Normalize points
                                hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                                hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                                hip_center_x = (hip_left.x + hip_right.x) / 2
                                hip_center_y = (hip_left.y + hip_right.y) / 2
                                hip_center_z = (hip_left.z + hip_right.z) / 2

                                scale = np.sqrt((hip_left.x - hip_right.x)**2 + (hip_left.y - hip_right.y)**2)
                                if scale < 1e-6: continue

                                normalized_landmarks = []
                                for landmark in LOWER_BODY_LANDMARKS:
                                    lm = landmarks[landmark.value]
                                    normalized_landmarks.append((lm.x - hip_center_x) / scale)
                                    normalized_landmarks.append((lm.y - hip_center_y) / scale)
                                    normalized_landmarks.append(lm.z - hip_center_z)

                                row = [video_id, class_name] + normalized_landmarks
                                writer.writerow(row)
                    cap.release()
    pose.close()
    print(f"\n✅ Processing finished. Data saved to {output_csv_path}")

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    DATA_FOLDER = project_root / 'data'
    OUTPUT_FOLDER = project_root / 'processed_data'
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    OUTPUT_CSV = OUTPUT_FOLDER / 'landmarks.csv'
    
    process_videos(DATA_FOLDER, OUTPUT_CSV)