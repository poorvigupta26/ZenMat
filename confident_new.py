import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import pyttsx3
import threading

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity = 0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained model and scaler
model_filename = 'yoga_pose_model_logistic.pkl'
scaler_filename = 'yoga_pose_scaler_logistic.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(scaler_filename, 'rb') as file:
    scaler = pickle.load(file)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

def speak(text):
    engine.say(text)
    engine.runAndWait()

def extract_landmarks(results):
    if results.pose_landmarks:
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return np.array(landmarks).flatten()
    return None

def predict_with_confidence(model, scaler, input_data):
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = np.max(probabilities)
    return prediction, confidence

#initializing variables for feedback mech
last_feedback_time = 0
last_feedback_confidence = 0
high_confidence_spoken = False

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


def capture_frames():
    global frame
    while cap.isOpened():
        ret, frame = cap.read()

threading.Thread(target=capture_frames, daemon=True).start()

while True:
 if frame is not None:
    frame = cv2.resize(frame, (640, 480))
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and get the landmarks
    results = pose.process(image)
    
    # Convert the image back to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Extract landmarks
    landmarks = extract_landmarks(results)
    
    if landmarks is not None:
        # Make prediction
        prediction, confidence = predict_with_confidence(model, scaler, landmarks)

        current_time = time.time()
        if confidence <= 0.5:
            if current_time - last_feedback_time >= 5:  # Check if 3 seconds have passed
                speak("Pose not right. Try again Please.")
                last_feedback_time = current_time
                last_feedback_confidence = confidence
                high_confidence_spoken = False
            elif confidence > last_feedback_confidence:
                speak("That's better! Keep improving.")
                last_feedback_time = current_time
                last_feedback_confidence = confidence
        elif 0.5 < confidence < 0.8:
            if current_time - last_feedback_time >= 5:  # Check if 3 seconds have passed
                speak("Good going. Keep trying!")
                last_feedback_time = current_time
                last_feedback_confidence = confidence
                high_confidence_spoken = False
            elif confidence > last_feedback_confidence:
                speak("You're improving! Keep it up.")
                last_feedback_time = current_time
                last_feedback_confidence = confidence
        else:  # This covers confidence >= 0.8
            if not high_confidence_spoken:
                speak("Congrats! You've got how to do this pose now.")
                high_confidence_spoken = True
                last_feedback_time = current_time
                last_feedback_confidence = confidence
            elif confidence < last_feedback_confidence:
                high_confidence_spoken = False  # Reset so it can speak again if confidence goes back up
        if confidence > last_feedback_confidence:
                last_feedback_confidence = confidence


        
        # Display prediction and confidence
        cv2.putText(image, f"Pose: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw pose landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display the image
    cv2.imshow('Yoga Pose Classification', image)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()