import cv2
import mediapipe as mp
import csv
import os

# --- CONFIGURATION ---
# Add the signs you want to teach the computer here!
SIGNS = ["Hello", "Thank You", "I Love You"]
DATA_FILE = "hand_data.csv"
# ---------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Create the CSV file and write the header
# We have 21 points, each with an X and Y coordinate = 42 columns + 1 for the Label
header = []
for i in range(21):
    header.append(f"x{i}")
    header.append(f"y{i}")
header.append("label")

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

current_sign_index = 0

print("--- DATA COLLECTION MODE ---")
print(f"Signs to collect: {SIGNS}")
print("Press 's' to start recording the current sign")
print("Press 'n' to move to the next sign")
print("Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success: break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Display which sign we are currently recording
    cv2.putText(img, f"Current Sign: {SIGNS[current_sign_index]}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "Press 's' to Save Frame | 'n' for Next Sign | 'q' to Quit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # If 's' is pressed, save the coordinates of the landmarks
            if cv2.waitKey(1) & 0xFF == ord('s'):
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                # Add the label at the end
                landmarks.append(SIGNS[current_sign_index])

                with open(DATA_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks)

                print(f"Saved one frame for {SIGNS[current_sign_index]}!")

    cv2.imshow("Data Collection", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_sign_index = (current_sign_index + 1) % len(SIGNS)
        print(f"Switched to sign: {SIGNS[current_sign_index]}")

cap.release()
cv2.destroyAllWindows()