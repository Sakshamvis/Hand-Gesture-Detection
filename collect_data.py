import cv2
import mediapipe as mp
import csv
import os

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- Paths ----------------
DATA_DIR = "../data"

GESTURES = {
    'h': 'hello',
    'i': 'iloveyou',
    'y': 'yes',
    'n': 'no',
    't': 'thankyou',
    'p': 'please'
}

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)

print("==== DATA COLLECTION MODE ====")
print("Press key to save sample:")
print("h → hello")
print("i → iloveyou")
print("y → yes")
print("n → no")
print("t → thankyou")
print("p → please")
print("q → quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0].landmark

        # Draw bounding box only
        h, w, _ = frame.shape
        x_vals = [int(lm.x * w) for lm in hand_landmarks]
        y_vals = [int(lm.y * h) for lm in hand_landmarks]
        x1, y1 = min(x_vals), min(y_vals)
        x2, y2 = max(x_vals), max(y_vals)

        cv2.rectangle(frame, (x1-20, y1-20), (x2+20, y2+20), (0,255,0), 2)

    cv2.imshow("Collect Gesture Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if chr(key) in GESTURES and result.multi_hand_landmarks:
        gesture_name = GESTURES[chr(key)]
        file_path = os.path.join(DATA_DIR, f"{gesture_name}.csv")

        row = []
        for lm in hand_landmarks:
            row.extend([lm.x, lm.y, lm.z])

        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"Saved sample for {gesture_name}")

cap.release()
cv2.destroyAllWindows()
