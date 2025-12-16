import cv2
import mediapipe as mp 
import numpy as np
import joblib 
import os

# load model
MODEL_PATH="../models/gesture_model.pkl"
model=joblib.load(MODEL_PATH)
# Label map (must match training)
LABELS = {
    0: "Hello ",
    1: "I Love You ",
    2: "Yes ",
    3: "No ",
    4: "Thank You ",
    5: "Please "
}

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_text = ""

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark

        # Extract landmarks into feature vector
        row = []
        for point in lm:
            row.extend([point.x, point.y, point.z])

        X = np.array(row).reshape(1, -1)

        # Predict
        prediction = model.predict(X)[0]
        confidence = np.max(model.predict_proba(X)) * 100

        gesture_text = f"{LABELS[prediction]}  ({confidence:.1f}%)"

        # Draw bounding box
        h, w, _ = frame.shape
        x_vals = [int(p.x * w) for p in lm]
        y_vals = [int(p.y * h) for p in lm]
        x1, y1 = min(x_vals), min(y_vals)
        x2, y2 = max(x_vals), max(y_vals)

        cv2.rectangle(frame, (x1-20, y1-20), (x2+20, y2+20), (0,255,0), 2)

    # Display text
    cv2.putText(
        frame,
        gesture_text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3
    )

    cv2.imshow("Hand Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()