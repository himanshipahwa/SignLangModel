import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
from tensorflow.keras.models import load_model

# 1. Initialize Voice
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 2. Load the Movement Brain
print("Loading the movement brain...")
model = load_model("movement_model.h5")

# 3. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- CONFIGURATION ---
# These MUST match EXACTLY what was trained in train_movement_model.py
SIGNS = [
    "I", "You", "Help", "Please", "Thank You",
    "Sorry", "Yes", "No", "Want", "Go",
    "Eat", "Like", "Need", "Know",
    "Good", "Bad", "Where", "Who", "What"
]
SEQUENCE_LENGTH = 30  # The "flip-book" size
# ---------------------

cap = cv2.VideoCapture(0)

# Buffers
sequence = [] # This is our "conveyor belt"
sentence = [] # This is our "notebook"
last_detected_sign = ""

print("Starting the Ultimate Sign App...")
print("Controls: [SPACE] Speak Sentence | [C] Clear Sentence | [Q] Quit")

while True:
    success, img = cap.read()
    if not success: break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            
            # Add the current frame to our conveyor belt (buffer)
            sequence.append(landmarks)
            
            # If the belt gets too long, remove the oldest frame
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            # Once we have a full 30-frame clip, we can predict!
            if len(sequence) == SEQUENCE_LENGTH:
                # Prepare the data for the brain
                input_data = np.expand_dims(sequence, axis=0)
                prediction = model.predict(input_data, verbose=0)
                
                # Get the sign with the highest probability
                sign_index = np.argmax(prediction)
                sign = SIGNS[sign_index]
                
                # Only add to sentence if it's a new sign
                if sign != last_detected_sign:
                    sentence.append(sign)
                    last_detected_sign = sign
                    print(f"Detected: {sign}")

    # --- UI DISPLAY ---
    # Show what is currently being detected
    cv2.putText(img, f"Detecting: {last_detected_sign}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the sentence at the bottom
    full_text = " ".join(sentence)
    cv2.rectangle(img, (0, 350), (640, 480), (0, 0, 0), -1)
    cv2.putText(img, f"Sentence: {full_text}", (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("AI Sign Language Translator", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if sentence:
            final_text = " ".join(sentence)
            engine.say(final_text)
            engine.runAndWait()
    elif key == ord('c'):
        sentence = []
        last_detected_sign = ""
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
