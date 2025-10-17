import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
from pathlib import Path
import json

# --- Adjustable parameters ---
SEQUENCE_LENGTH = 30       # Must be the same as in the training script
CONFIDENCE_THRESHOLD = 0.9 # Minimum model confidence to show prediction

def main():
    # Loading paths and models
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / 'models' / 'step_classifier.h5'
    labels_path = project_root / 'processed_data' / 'label_map.json'

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    with open(labels_path, 'r') as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(len(label_map))]
    print(f"Model loaded. Classes: {class_names}")

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Definition of keypoints (must be identical to 01_process_data.py)
    LOWER_BODY_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    ]

    # Queue for saving sequences
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    current_prediction = ""

    # Launch the camera
    cap = cv2.VideoCapture(0) # 0 is for the default webcam
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Image processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # Returning the display color
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Drawing a skeleton
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            landmarks = results.pose_landmarks.landmark
            all_landmarks_visible = all(landmarks[lm.value].visibility > 0.6 for lm in LOWER_BODY_LANDMARKS)

            if all_landmarks_visible:
                # Normalization (same as in 01_process_data.py)
                hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                hip_center_x = (hip_left.x + hip_right.x) / 2
                hip_center_y = (hip_left.y + hip_right.y) / 2
                hip_center_z = (hip_left.z + hip_right.z) / 2
                scale = np.sqrt((hip_left.x - hip_right.x)**2 + (hip_left.y - hip_right.y)**2)
                
                if scale > 1e-6:
                    normalized_landmarks = []
                    for landmark in LOWER_BODY_LANDMARKS:
                        lm = landmarks[landmark.value]
                        normalized_landmarks.append((lm.x - hip_center_x) / scale)
                        normalized_landmarks.append((lm.y - hip_center_y) / scale)
                        normalized_landmarks.append(lm.z - hip_center_z)
                    
                    sequence.append(normalized_landmarks)

                    # If the sequence is full, we make a prediction
                    if len(sequence) == SEQUENCE_LENGTH:
                        input_data = np.expand_dims(np.array(sequence), axis=0) # Add a batch dimension
                        
                        prediction = model.predict(input_data)[0]
                        confidence = np.max(prediction)
                        
                        if confidence > CONFIDENCE_THRESHOLD:
                            predicted_class_index = np.argmax(prediction)
                            current_prediction = class_names[predicted_class_index]
                        
        # Displaying the prediction on the screen
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, current_prediction.upper(), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Showing video
        cv2.imshow('Folklor AI', image)

        # Quit with 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    main()