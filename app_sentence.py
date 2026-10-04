import cv2
import mediapipe as mp
import pickle
import pyttsx3
import numpy as np

# 1. Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 2. Load the trained ML Brain
print("Loading the brain...")
with open("model.p", "rb") as f:
    model = pickle.load(f)

# 3. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- SENTENCE BUILDING VARIABLES ---
sentence = []
last_detected_sign = ""
# -----------------------------------

print("Starting Sentence Builder... ")
print("Controls:")
print("- Make signs to add words to the sentence")
print("- Press 'SPACE' to speak the full sentence")
print("- Press 'c' to clear the sentence")
print("- Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success: break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            prediction = model.predict([landmarks])
            sign = prediction[0]

            # If this is a new sign, add it to our sentence
            if sign != last_detected_sign:
                sentence.append(sign)
                last_detected_sign = sign
                print(f"Added to sentence: {sign}")

    # --- UI DISPLAY ---
    # Show the current sign being detected
    cv2.putText(img, f"Detecting: {last_detected_sign}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the full sentence being built at the bottom
    full_text = " ".join(sentence)
    cv2.rectangle(img, (0, 350), (640, 480), (0, 0, 0), -1) # Black background for text
    cv2.putText(img, f"Sentence: {full_text}", (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(img, "Space: Speak | C: Clear | Q: Quit", (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Sign Language Sentence Builder", img)

    key = cv2.waitKey(1) & 0xFF

    # Press SPACE to speak the sentence
    if key == ord(' '):
        if sentence:
            final_sentence = " ".join(sentence)
            print(f"Speaking: {final_sentence}")
            engine.say(final_sentence)
            engine.runAndWait()
        else:
            print("Sentence is empty!")

    # Press 'c' to clear the sentence
    elif key == ord('c'):
        sentence = []
        last_detected_sign = ""
        print("Sentence cleared.")

    # Press 'q' to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()