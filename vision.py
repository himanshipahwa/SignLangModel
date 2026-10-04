import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Hand Tracking
# This tells the computer we want to track hands specifically
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 2. Start the Camera
# '0' usually refers to your built-in webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit the window")

while True:
    # Capture a single frame (picture) from the camera
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # MediaPipe needs colors in RGB, but OpenCV uses BGR. We have to swap them!
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and find hand landmarks
    results = hands.process(img_rgb)

    # If hands are found, draw the dots (landmarks) and lines (connections)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the final result in a window
    cv2.imshow("Sign Language Vision", img)

    # Stop the program if you press the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: Turn off the camera and close the window
cap.release()
cv2.destroyAllWindows()