# realtime_interpreter.py
# Run this after training your model with train_model.py

import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ---------- Load Model ----------
model = joblib.load("model/sign_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ---------- Label Mapping ----------
# These must match the labels you used in collect_data.py
pretty_labels = {
    "THUMBS_UP": "Thumbs Up 👍",
    "LOVE": "Love ❤️",
    "OK": "OK 👌",
    "STOP": "Stop ✋",
    "CALL": "Call Me 🤙"
}

# ---------- Mediapipe Setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def extract_hand_features(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    rel = pts - wrist
    size = np.linalg.norm(pts[9] - pts[0])
    if size == 0:
        size = 1e-6
    normalized = rel.flatten() / size
    return normalized

# ---------- Start Webcam ----------
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=1
) as hands:
    print("Showing predictions in real time... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                features = extract_hand_features(lm.landmark)
                features = scaler.transform([features])
                pred = model.predict(features)[0]
                prob = model.predict_proba(features).max()

                # Convert label to nice readable format
                display_text = f"{pretty_labels.get(pred, pred)} ({prob:.2f})"

                # Show prediction on the frame
                cv2.putText(frame, display_text, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Real-Time Sign Language Interpreter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
