# Real-time-sign-language-interpreter

A **real-time sign language interpreter** using Python, OpenCV, and MediaPipe that can detect multiple hand gestures and display their names on the screen. This project can help bridge communication between people who use sign language and those who don’t understand it.

---

## Features
- Real-time hand gesture recognition using a webcam
- Supports multiple gestures: 
  - 👍 Thumbs Up  
  - 👎 Thumbs Down  
  - ✋ Stop  
  - 👌 OK  
  - ✊ Fist  
  - 🤙 Call Me  
  - ✌️ Love / Peace  
  - 👉 Point  
- Displays gesture names with confidence on video feed
- Modular design:
  - `collect_data.py` → Capture your gestures
  - `train_model.py` → Train your classifier
  - `realtime_interpreter.py` → Run real-time interpreter

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Thol-444/real-time-sign-language-interpreter.git
cd real-time-sign-language-interpreter
