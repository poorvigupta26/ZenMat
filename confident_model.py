import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.8)

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return np.array(landmarks).flatten()
    return None

def load_dataset(dataset_path, split):
    X = []
    y = []
    
    split_path = os.path.join(dataset_path, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = cv2.imread(image_path)
                
                if image is not None:
                    landmarks = extract_landmarks(image)
                    if landmarks is not None:
                        X.append(landmarks)
                        y.append(class_name)
    
    return np.array(X), np.array(y)

# Load and prepare your dataset
dataset_path = "C:\\Users\\Dell\\OneDrive\\Desktop\\SIH\\yoga_poses"  # Replace with the actual path to your dataset
X_train, y_train = load_dataset(dataset_path, "train")
X_test, y_test = load_dataset(dataset_path, "test")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
classes = ['chair', 'cobra', 'dog', 'no_pose', 'shoulder_stand', 'traingle', 'tree', 'warrior']  # replace with your actual pose names
model = LogisticRegression(multi_class='ovr', max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Evaluate the model
accuracy = model.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy}")

# Example prediction with confidence scores
example_image_path = "path_to_some_test_image.jpg"  # Replace with actual image path
example_image = cv2.imread(example_image_path)
if example_image is not None:
    example_landmarks = extract_landmarks(example_image)
    if example_landmarks is not None:
        example_landmarks_scaled = scaler.transform([example_landmarks])
        prediction = model.predict(example_landmarks_scaled)
        confidence_scores = model.predict_proba(example_landmarks_scaled)
        print(f"Predicted pose: {prediction[0]}")
        print(f"Confidence scores: {confidence_scores[0]}")
    else:
        print("No landmarks detected in the image.")
else:
    print("Could not read the image.")

# Save the model and scaler as pickle files
model_filename = 'yoga_pose_model_logistic.pkl'
scaler_filename = 'yoga_pose_scaler_logistic.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)

print(f"Model saved as {model_filename}")
print(f"Scaler saved as {scaler_filename}")
