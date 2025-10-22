# collect_data.py
# Usage:
# 1) Run: python collect_data.py
# 2) Press one of the keys (t, l, o, s, c) to select a gesture label.
# 3) Press 's' to start/stop recording for that gesture.
# 4) Press 'q' to quit.

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os
from collections import deque

mp_hands = mp.solutions.hands
os.makedirs('data', exist_ok=True)
csv_file = 'data/sign_data.csv'

# If file doesn't exist create header later
header_written = os.path.exists(csv_file)

def extract_hand_features(landmarks):
    # landmarks: list of 21 landmarks, each has x,y,z normalized to image
    # Convert to relative coords (w.r.t. wrist) and normalize by hand size.
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    rel = pts - wrist
    # estimate hand size using distance between wrist and middle finger MCP
    size = np.linalg.norm(pts[9] - pts[0])  # using index 9 (middle finger PIP/MCP region)
    if size == 0:
        size = 1e-6
    normalized = rel.flatten() / size
    return normalized.tolist()  # list length 63

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
    print("Press a key to select gesture:")
    print("t=THUMBS_UP, l=LOVE, o=OK, s=STOP, c=CALL")
    print("Press 's' to start/stop recording, 'q' to quit.")
    
    recording = False
    current_label = None
    buffer = deque(maxlen=30)  # for smoothing

    # Map shortcut keys to gesture names
    label_map = {
        't': 'THUMBS_UP',
        'd': 'THUMBS_DOWN',
        's': 'STOP',
        'o': 'OK',
        'f': 'FIST',
        'c': 'CALL',
        'l': 'LOVE',
        'p': 'POINT'
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            for landmark in lm.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 2, (0,255,0), -1)
            features = extract_hand_features(lm.landmark)
            buffer.append(features)
        else:
            features = None

        info = f"Label: {current_label} | Recording: {recording} | Buffer: {len(buffer)}"
        cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Collect Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key != 255:
            ch = chr(key).lower()
            if ch == 'q':
                break
            elif ch == 's':  # toggle recording for current label
                if current_label is None:
                    print("Choose a label first by pressing one of:", list(label_map.keys()))
                else:
                    recording = not recording
                    print("Recording:", recording, "label:", current_label)
                    if recording:
                        time.sleep(0.5)
            elif ch in label_map:
                current_label = label_map[ch]
                print("Current label set to:", current_label)
            else:
                print("Unknown key. Use one of:", list(label_map.keys()))

        # If recording and buffer has frames, write them to CSV
        if recording and len(buffer) > 0:
            feat = buffer[-1]
            row = feat + [current_label]
            write_header = False
            if not header_written:
                write_header = True
                header_written = True
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    header = []
                    for i in range(21):
                        header += [f'x{i}', f'y{i}', f'z{i}']
                    header += ['label']
                    writer.writerow(header)
                writer.writerow(row)
            cv2.putText(frame, "SAVED", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            time.sleep(0.15)

cap.release()
cv2.destroyAllWindows()
