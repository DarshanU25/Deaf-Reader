# ✋ ISL/ASL Hand Gesture Recognition (Single & Two-Hand)

**Note:** You will need to run the `requirements.txt` file to install all dependencies and use **Python 3.8**.  
The `requirements.txt` will also create a virtual environment for you.

---

A Python-based Hand Gesture Recognition system for **Indian Sign Language (ISL)** and **American Sign Language (ASL)**.  
Supports **single-hand** and **two-hand** gestures, includes a **custom Mediapipe landmark logger** to CSV, **separate training pipelines**, and **live webcam detection**.

---

## 📌 Features
- Real-time detection for **1-hand** and **2-hand** gestures.
- **Custom landmark logger**: saves Mediapipe hand coordinates to CSV.
- Separate training scripts for **single-hand** and **two-hand** models.
- **Live prediction** from webcam after training.

---

## 🚀 Quick Start

### 0) Install Dependencies
Run:
```bash
bash requirements.txt

### 1) Collect Landmark Coordinates
Run:
    python Custom_landmark_detect.py

Controls:
- Press **k** → starts saving landmark values when a hand is detected; if not detected, logs 0s.
- Press **a, b, c, ... z** → set/change the current gesture class label.
- Press **ESC** → exit.

Output file:
    logged_data.csv

---

### 2) Clean & Organize Data
- Open **logged_data.csv** and remove any **wrong/noisy rows** (e.g., poor detections).
- If your gesture uses **two hands**:
  - Copy cleaned data to:
        model/keypoint_classifier/keypoint_twohand.csv
- If your gesture uses **one hand**:
  - Copy cleaned data to:
        model/keypoint_classifier/keypoint_singlehand.csv

---

### 3) Train the Models
Single-hand model:
    python train_model_singlehand.py

Two-hand model:
    python train_model_twohand.py

---

### 4) Live Gesture Detection
After training, run:
    python live_detection.py

---

## 📂 Project Structure
    ├── Custom_landmark_detect.py               # Collect & log Mediapipe hand landmarks to CSV
    ├── live_detection.py                       # Live webcam prediction
    ├── train_model_singlehand.py               # Train single-hand gesture model
    ├── train_model_twohand.py                  # Train two-hand gesture model
    ├── logged_data.csv                         # Raw collected landmarks (to be cleaned)
    └── model/
        └── keypoint_classifier/
            ├── keypoint_singlehand.csv         # Cleaned single-hand training data
            └── keypoint_twohand.csv            # Cleaned two-hand training data

---

## 🛠️ Notes & Tips
- Ensure consistent **class labels** while logging (use a–z keys).
- Remove obviously wrong detections before moving data to the model CSVs.
- Keep single-hand and two-hand samples **in their respective CSV files**.

---

## 📜 License
MIT License
