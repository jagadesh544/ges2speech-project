import cv2
import numpy as np
import mediapipe as mp
import pickle
import pyttsx3
import os
import time
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture Labels
GESTURES = ["Hello", "Love You", "Thumbs Up", "Rock", "Bye", "call me", "good luck", "I want to talk", "loser", "ok", "Victory", "Hurts a lot", "Good job", "My name is", "Help", "please", "stop", "Eat", "More", "What", "Where", "when"]

# Model Files
MODEL_FILE = "gesture_model.pkl"
DATA_FILE = "gesture_data.pkl"

def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(DATA_FILE):
        with open(MODEL_FILE, "rb") as f:
            knn = pickle.load(f)
        with open(DATA_FILE, "rb") as f:
            gesture_data = pickle.load(f)
        return knn, gesture_data
    return None, None

# Load model
knn, gesture_data = load_model()
engine = pyttsx3.init()

# Streamlit UI
st.title("Gesture to Speech Recognition")
st.write("Click 'Start Detection' to begin.")

# Initialize session state
if "start_detection" not in st.session_state:
    st.session_state.start_detection = False

# Buttons
if st.button("Start Detection"):
    st.session_state.start_detection = True

if st.button("Stop Detection"):
    st.session_state.start_detection = False

frame_placeholder = st.empty()
message_placeholder = st.empty()

# Run detection if button clicked
if st.session_state.start_detection and knn:
    cap = cv2.VideoCapture(0)
    last_spoken = None
    time.sleep(2)  # 2-second delay before detection starts
    
    while st.session_state.start_detection:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_data = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                
                if gesture_data and all(gesture_data.values()) and hand_data.shape[0] == len(next(iter(gesture_data.values()))[0]):
                    prediction = knn.predict([hand_data])[0]
                    gesture_name = GESTURES[prediction]
                    
                    if gesture_name != last_spoken:
                        last_spoken = gesture_name
                        message_placeholder.write(f"Detected Gesture: {gesture_name}")
                        engine.say(gesture_name)
                        engine.runAndWait()
        
        frame_placeholder.image(frame, channels="BGR")
    
    cap.release()
