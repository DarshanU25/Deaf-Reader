import cv2 as cv
import csv
import copy
import time
import numpy as np
import mediapipe as mp

# ======= Utility Functions =======

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    normalized = [(x - base_x, y - base_y) for x, y in landmark_list]
    max_val = max([abs(val) for pair in normalized for val in pair])
    return [coord / max_val if max_val else 0 for pair in normalized for coord in pair]

def calc_landmark_list(image, landmarks):
    h, w, _ = image.shape
    return [(int(point.x * w), int(point.y * h)) for point in landmarks.landmark]

def select_mode(key, mode):
    number = -1
    if 97 <= key <= 122:       # 'a' to 'z' → 1–26
      number = key - 96
    elif key == 32:            # Space → 27
      number = 27
    elif key == 8:             # Backspace/Delete → 28
      number = 28
    elif key == 13:            # Enter/Confirm → 29
      number = 29


    if key == ord('k'):
        mode = 1
        print("📝 Logging started.")
    elif key == ord('s'):
        mode = 0
        print("⛔ Logging stopped.")

    return number, mode

# ======= Main Program =======

def main():
    cap = cv.VideoCapture(0)  # Change to 0 if you use the built-in webcam
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    csv_path = 'logged_data.csv'
    number = -1
    mode = 0
    frame_interval = 1 / 3  # 3 frames per second
    last_logged = time.time()

    print("ℹ️ Press 'k' to start logging, 's' to stop, 'Esc' to quit.")

    while True:
        ret, image = cap.read()
        if not ret:
            break

        key = cv.waitKey(1)
        if key == 27:  # ESC to exit
            break
        if key != -1:
            selected_number, mode = select_mode(key, mode)
            if selected_number != -1:
                number = selected_number
                print(f"✅ Gesture label selected: {number}")

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        landmark_list_combined = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 👇 Draw skeleton
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                landmark_list_combined.append(landmark_list)

        # Ensure always 2 hands worth of data
        while len(landmark_list_combined) < 2:
            landmark_list_combined.append([[0, 0]] * 21)

        # Logging logic
        current_time = time.time()
        if mode == 1 and number != -1 and current_time - last_logged >= frame_interval:
            flat_data = []
            for hand in landmark_list_combined[:2]:
                processed = pre_process_landmark(hand)
                flat_data.extend(processed)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([number] + flat_data)
            last_logged = current_time
            print("📸 Logged frame")

        cv.imshow('ISL Logger', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
