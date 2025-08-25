# Real-Time AI Yoga Pose Correction System

![Yoga Coach Demo](https-placeholder-for-your-gif-or-video-link.gif)
*(Space for Video Demo - Replace the above line with a GIF or a link to a video of your project in action.)*

---

## 📖 Overview

This project is a real-time, AI-powered virtual yoga coach that uses computer vision to analyze a user's yoga posture and provide immediate, actionable feedback. By leveraging a standard webcam, the system aims to make personalized yoga instruction more accessible, helping users improve their form, maximize benefits, and reduce the risk of injury.

A key innovation of this project is its **view-segmented** approach. The underlying deep learning model is trained on a custom dataset that includes front, back, and side views for each yoga pose, allowing for more accurate and context-aware posture correction.

The system provides a multi-modal feedback loop, delivering:
1.  **Pose & View Classification:** Identifies the yoga pose and the user's viewing angle.
2.  **Quantitative Correctness Score:** A percentage-based score (0-100%) calculated using Cosine Similarity against an ideal pose template.
3.  **Qualitative, Model-Driven Feedback:** Specific, interpretable instructions pinpointing which body parts (e.g., "ARMS", "LEGS") require correction, derived using an explainable AI technique.

> **⚠️ Disclaimer:** This is an academic and experimental project currently under development. The feedback provided should be considered as a supplementary guide and not a replacement for professional medical advice or instruction from a certified yoga teacher.

---

## 🛠️ System Architecture

The project is architected into two distinct pipelines: an **Offline Training Pipeline** for building the AI model and an **Online Inference Pipeline** for the real-time application.

### 1. Offline Training Pipeline
This pipeline processes raw video data to create the machine learning artifacts needed for the real-time coach.

1.  **Data Acquisition:** A custom video dataset is created, with videos for each yoga asana categorized by correctness (`Right`/`Wrong`) and view (`Front`/`Back`/`Side`).
2.  **Feature Extraction (MediaPipe):** The script `1_preprocess_data.py` processes each video frame-by-frame. It uses Google's MediaPipe Pose to extract 33 normalized 3D landmarks for the detected person.
3.  **Vectorization:** These landmarks are flattened into a 132-dimensional feature vector (`33 landmarks * 4 coordinates (x, y, z, visibility)`).
4.  **Dataset Creation:** The labeled vectors are compiled into a single structured dataset (`yoga_landmarks.csv`).
5.  **Model Training:** The script `2_train_model.py` uses the CSV to train a Dense Neural Network (DNN) with TensorFlow/Keras. The model learns to classify the fine-grained pose labels (e.g., `Tadasana_Right_Front`). The final trained model (`.h5`) and a label encoder (`.pkl`) are saved.

### 2. Online Real-Time Inference Pipeline
This is the live application that the user interacts with (`3_realtime_coach.py`).

1.  **Live Pose Estimation:** Captures frames from the webcam and uses MediaPipe to generate a live 132-dimensional pose vector.
2.  **Multi-Modal Evaluation:** For each frame, the system performs three parallel evaluations:
    *   **DNN Classification:** The pose vector is fed to the trained DNN to get a primary prediction (e.g., `Vajrasana_Wrong_Side`).
    *   **Geometric Scoring:** The vector is compared to a pre-calculated "ideal" pose template using **Cosine Similarity** to generate a quantitative SCORE.
    *   **Heuristic Feedback:** If the pose is classified as "Wrong" or the score is low, a perturbation-based analysis is performed to identify which body parts (ARMS, LEGS, TORSO) are the main contributors to the error.
3.  **Visualization:** All three outputs are synthesized and rendered on the live video feed, providing the user with comprehensive, real-time feedback.

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

Make sure you have Python 3.8+ and the following libraries installed:

```bash
pip install tensorflow opencv-python mediapipe scikit-learn pandas
```

### 1. Prepare the Dataset

-   Create a main folder named `Videos`.
-   Inside `Videos`, create a subfolder for each yoga pose (e.g., `Tadasana`, `Vajrasana`).
-   Inside each pose folder, create subfolders for each correctness and view combination, named in the format `Correctness_View` (e.g., `Right_Front`, `Wrong_Side`).
-   Place your short, trimmed video clips of the final held poses into the corresponding folders.

### 2. Run the Pipeline

Execute the scripts in the following order from your terminal:

**Step 1: Preprocess the Data**
This will read your videos and create the `yoga_landmarks.csv` file.
```bash
python 1_preprocess_data.py
```

**Step 2: Train the Model**
This will use the CSV to train the DNN and save the `.h5` and `.pkl` files.
```bash
python 2_train_model.py
```

**Step 3: Launch the Real-Time Coach**
This will start your webcam and the live feedback application.
```bash
python 3_realtime_coach.py
```
Press 'q' to quit the application.

---

## 🔮 Future Work

This project serves as a strong foundation for several exciting future enhancements:

-   **Dataset Expansion:** The highest priority is to expand the dataset to include more yoga poses and, crucially, videos of multiple subjects with diverse body types to improve generalization.
-   **Stability Score:** Implement a feature to analyze pose stability by measuring landmark deviation over a time window, providing feedback on wobbling.
-   **Transition Analysis:** Upgrade the model architecture (e.g., using LSTMs or Transformers) to analyze the quality of movement *between* poses.
-   **"Neutral" Class:** Add a "Neutral" class (e.g., sitting, standing) to the dataset to make the system more robust when the user is not actively performing a pose.
-   **Mobile Deployment:** Port the system to a mobile application for greater accessibility.

---

##  MENTIONS

I extend my sincere gratitude to my professor, **Srimanta Mandal**, for their invaluable guidance, insightful feedback, and constant encouragement throughout the development of this project. Their expertise was instrumental in shaping the methodology and overcoming challenges.