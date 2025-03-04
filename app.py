import cv2
import numpy as np
import mediapipe as mp
import pickle
import pyttsx3
import os
from sklearn.neighbors import KNeighborsClassifier

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Gesture Labels
GESTURES = ["Hello", "Love You", "Thumbs Up", "Zero", "Bye","call me","good luck","I want to talk","loser","ok","Victory","Hurts a lot","Good job"]

# Check if a trained model exists
MODEL_FILE = "gesture_model.pkl"
DATA_FILE = "gesture_data.pkl"
gesture_data = {gesture: [] for gesture in GESTURES}

# Text-to-Speech Engine
engine = pyttsx3.init()

# Function to extract hand landmarks
def extract_hand_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

# Check if previous data exists
use_previous = False
if os.path.exists(MODEL_FILE) and os.path.exists(DATA_FILE):
    choice = input("Use previously trained gestures? (yes/no): ").strip().lower()
    if choice == "yes":
        use_previous = True

# Load previous trained model and data
if use_previous:
    with open(MODEL_FILE, "rb") as f:
        knn = pickle.load(f)
    with open(DATA_FILE, "rb") as f:
        gesture_data = pickle.load(f)
    print("Loaded previous trained model.")
else:
    print("Retraining gestures...")

    # Capture Training Data
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture a sample for a gesture. Press 'q' to quit training.")

    for gesture in GESTURES:
        print(f"Show '{gesture}' gesture and press 'c' to capture (3 times).")
        samples = 0
        while samples < 3:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Show {gesture} - Captured: {samples}/3", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Gesture Capture", frame)

            key = cv2.waitKey(1)
            if key == ord('c') and results.multi_hand_landmarks:
                hand_data = extract_hand_landmarks(results.multi_hand_landmarks[0])
                gesture_data[gesture].append(hand_data)
                samples += 1
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cap.release()
    cv2.destroyAllWindows()

    # Merge previous and new training data
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            previous_data = pickle.load(f)
        for gesture in GESTURES:
            if gesture in previous_data:
                gesture_data[gesture].extend(previous_data[gesture])

    # Convert Data to NumPy Arrays
    X = []
    y = []
    for idx, (gesture, samples) in enumerate(gesture_data.items()):
        for sample in samples:
            X.append(sample)
            y.append(idx)

    X = np.array(X)
    y = np.array(y)

    # Train a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Save Model and Data
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(knn, f)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(gesture_data, f)

    print("Training complete. Starting real-time detection...")

# Start Gesture Recognition
cap = cv2.VideoCapture(0)

# Variable to track the last spoken gesture
last_spoken = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_data = extract_hand_landmarks(hand_landmarks)

            if hand_data.shape[0] == len(gesture_data[GESTURES[0]][0]):  # Ensure shape consistency
                prediction = knn.predict([hand_data])[0]
                gesture_name = GESTURES[prediction]

                # Check if the recognized gesture is different from the last spoken one
                if gesture_name != last_spoken:
                    # Update the last spoken gesture
                    last_spoken = gesture_name

                    # Display gesture name on screen
                    cv2.putText(frame, gesture_name, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Say the gesture name
                    engine.say(gesture_name)
                    engine.runAndWait()

    cv2.imshow("Gesture to Speech", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
