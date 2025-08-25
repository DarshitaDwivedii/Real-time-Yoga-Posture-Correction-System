import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# --- CORRECTED HELPER FUNCTIONS ---

def create_pose_templates(df):
    """
    CORRECTED: Creates an average landmark vector for each 'Right' pose VIEW.
    Returns a dictionary with keys like 'Tadasana_Front', 'Vajrasana_Side', etc.
    """
    pose_templates = {}
    right_labels = {label for label in df['label'].unique() if "_Right_" in label}
    
    for right_label in right_labels:
        parts = right_label.split('_')
        if len(parts) != 3: continue
        
        pose_name, _, view = parts
        template_key = f"{pose_name}_{view}" # e.g., 'Tadasana_Front'
        
        right_pose_df = df[df['label'] == right_label]
        if not right_pose_df.empty:
            pose_vector = right_pose_df.drop('label', axis=1).mean().values
            pose_templates[template_key] = pose_vector.reshape(1, -1)
            
    return pose_templates

def find_error_contributors(model, current_pose_vector, pose_templates, full_predicted_class, base_prediction_probs, label_encoder, predict_fn):
    """
    CORRECTED: Identifies which body parts are the primary contributors to an error.
    Works with the new 'PoseName_Correctness_View' label structure.
    """
    body_part_indices = {
        'ARMS': list(range(11, 23)),
        'LEGS': list(range(23, 33)),
        'TORSO': [11, 12, 23, 24]
    }
    
    parts = full_predicted_class.split('_')
    if len(parts) != 3: return ["Alignment"]
    pose_name, correctness, view = parts

    # We need the "Wrong" index and the "perfect" template for the *current view*.
    wrong_class_name = f"{pose_name}_Wrong_{view}"
    template_key = f"{pose_name}_{view}"
    
    try:
        wrong_class_idx = list(label_encoder.classes_).index(wrong_class_name)
    except ValueError:
        return ["Alignment"]

    template_vector = pose_templates.get(template_key)
    if template_vector is None: return ["Alignment"]

    base_wrong_confidence = base_prediction_probs[wrong_class_idx]
    contributors = []
    
    for part_name, indices in body_part_indices.items():
        modified_vector = np.copy(current_pose_vector)
        
        for i in indices:
            modified_vector[0, i] = template_vector[0, i]
            modified_vector[0, i + 33] = template_vector[0, i + 33]
            modified_vector[0, i + 66] = template_vector[0, i + 66]
            modified_vector[0, i + 99] = template_vector[0, i + 99]
            
        new_probs = predict_fn(modified_vector).numpy()[0]
        new_wrong_confidence = new_probs[wrong_class_idx]
        
        # We check the drop in confidence relative to the *specific* wrong class
        if base_wrong_confidence > 0 and (base_wrong_confidence - new_wrong_confidence) / base_wrong_confidence > 0.25: # Relative drop
            contributors.append(part_name)
            
    return contributors if contributors else ["Alignment"]

# --- Main Application ---

try:
    model = load_model("yoga_pose_classifier.h5")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading model or encoder: {e}\nPlease run training script first.")
    exit()

@tf.function
def predict_on_frame(input_data):
    return model(input_data, training=False)

try:
    landmarks_df = pd.read_csv("yoga_landmarks_segmented.csv") # Make sure you're loading the correct segmented CSV
    pose_templates = create_pose_templates(landmarks_df)
    print("Successfully created ground truth templates for:", list(pose_templates.keys()))
except FileNotFoundError:
    print("Error: 'yoga_landmarks_segmented.csv' not found. Please run the preprocessing script first.")
    exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = [lm.x for lm in landmarks] + [lm.y for lm in landmarks] + [lm.z for lm in landmarks] + [lm.visibility for lm in landmarks]
            X = np.array(row, dtype=np.float32).reshape(1, -1)
            
            prediction_probs = predict_on_frame(X).numpy()[0]
            predicted_class_idx = np.argmax(prediction_probs)
            confidence = prediction_probs[predicted_class_idx]
            full_predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
            
            parts = full_predicted_class.split('_')
            if len(parts) == 3:
                pose_name, correctness, view = parts
            else:
                pose_name, correctness, view = parts[0], "Unknown", "Unknown"
            
            is_correct = correctness == "Right"
            
            # CORRECTED: Calculate score using view-specific template
            similarity_score = 0.0
            template_key = f"{pose_name}_{view}"
            if template_key in pose_templates:
                ground_truth_vector = pose_templates[template_key]
                similarity = cosine_similarity(X, ground_truth_vector)
                similarity_score = max(0, min(100, similarity[0][0] * 100))

            # CORRECTED: Feedback logic
            if is_correct and similarity_score > 90:
                feedback = "Excellent Form!"
            else:
                contributors = find_error_contributors(model, X, pose_templates, full_predicted_class, prediction_probs, label_encoder, predict_on_frame)
                feedback = "Improve: " + ", ".join(contributors)

            # --- VISUALIZATION (Updated to show view) ---
            pose_color = (0, 255, 0) if is_correct else (0, 0, 255)
            display_text = f"POSE: {pose_name} ({view})"
            cv2.putText(image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, pose_color, 2, cv2.LINE_AA)
            cv2.putText(image, f"CONF: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            score_color = (0, 255, 0) if similarity_score > 90 else (0, 255, 255) if similarity_score > 75 else (0, 0, 255)
            cv2.putText(image, f"SCORE: {similarity_score:.1f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2, cv2.LINE_AA)

            feedback_color = (0, 255, 0) if feedback == "Excellent Form!" else (0, 165, 255)
            cv2.putText(image, f"FEEDBACK: {feedback}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2, cv2.LINE_AA)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
    except Exception as e:
        print(f"An error occurred during real-time loop: {e}")
        pass

    cv2.imshow('Real-time Yoga Coach', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()