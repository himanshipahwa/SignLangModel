import cv2
import mediapipe as mp
import numpy as np
import os

# --- CONFIGURATION ---
# Add the signs that involve MOVEMENT here!
SIGNS = [
    "I", "You", "Help", "Please", "Thank You", 
    "Sorry", "Yes", "No", "Want", "Go", 
    "Eat", "Like", "Need", "Know",  
    "Good", "Bad", "Where", "Who", "What"
]
SEQUENCE_LENGTH = 30 # We will record 30 frames per sign
DATA_PATH = "MP_Data" # We will save data in a folder, not a CSV
# ---------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create the folder to save the data (always ensure all sign folders exist)
os.makedirs(DATA_PATH, exist_ok=True)
for sign in SIGNS:
    os.makedirs(os.path.join(DATA_PATH, sign), exist_ok=True)


cap = cv2.VideoCapture(0)
current_sign_index = 0

print("--- MOVEMENT DATA COLLECTION ---")
print(f"Signs: {SIGNS}")
print("Press 's' to record a 1-second clip | 'n' for Next Sign | 'q' to Quit")

while True:
    success, img = cap.read()
    if not success: break
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    cv2.putText(img, f"Current Sign: {SIGNS[current_sign_index]}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Movement Collection", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        current_sign_index = (current_sign_index + 1) % len(SIGNS)
    elif key == ord('s'):
        # --- THE MAGIC PART: RECORDING A SEQUENCE ---
        print(f"Recording {SIGNS[current_sign_index]}... Keep moving your hand!")
        sequence_data = []
        
        # Record 30 frames in a row
        for frame_num in range(SEQUENCE_LENGTH):
            # We need to capture a new frame for every single point in the sequence
            # Note: In a real app we'd use a buffer, but for baby steps, we'll capture now
            success, img_rec = cap.read()
            img_rgb_rec = cv2.cvtColor(img_rec, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb_rec)
            
            if res.multi_hand_landmarks:
                # Get the 21 points (x, y)
                landmarks = []
                for lm in res.multi_hand_landmarks[0].landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                sequence_data.append(landmarks)
            else:
                # If hand is lost during recording, fill with zeros
                sequence_data.append([0] * 42)
        
        # Save the 30 frames as a numpy file (special format for AI)
        # We create a unique filename for each clip (e.g., thank_you_0.npy)
        folder = os.path.join(DATA_PATH, SIGNS[current_sign_index])
        num_files = len(os.listdir(folder))
        file_path = os.path.join(folder, f"{num_files}.npy")
        
        np.save(file_path, np.array(sequence_data))
        print(f"Saved clip {num_files} for {SIGNS[current_sign_index]}!")

cap.release()
cv2.destroyAllWindows()