import cv2
import mediapipe as mp
import pickle
import pyttsx3
import numpy as np

# 1. Initialize Text-to-Speech (The Voice)
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech

# 2. Load the trained ML Brain
print("Loading the brain...")
with open("model.p", "rb") as f:
    model = pickle.load(f)

# 3. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# This variable prevents the computer from screaming the same word 30 times per second
last_spoken_sign = ""

print("Starting Application... Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success: break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the landmarks for the model
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Predict the sign
            # The model expects a 2D array, so we put our list inside another list
            prediction = model.predict([landmarks])
            sign = prediction[0]

            # Display the prediction on screen in big green letters
            cv2.putText(img, f"Sign: {sign}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # SPEAK the sign if it's different from the last one
            if sign != last_spoken_sign:
                print(f"Speaking: {sign}")
                engine.say(sign)
                engine.runAndWait()
                last_spoken_sign = sign

    cv2.imshow("Sign Language to Speech", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()