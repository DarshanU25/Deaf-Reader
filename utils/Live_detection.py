import cv2
import numpy as np
import mediapipe as mp
import time
import tensorflow as tf

class DualHandGestureClassifier:
    def __init__(self, single_hand_model_path, two_hand_model_path):
        # This is the constructor. It runs when you do: DualHandGestureClassifier(...)
        # Load single-hand TFLite model
        self.single_interpreter = tf.lite.Interpreter(model_path=single_hand_model_path)
        self.single_interpreter.allocate_tensors()
        self.single_input = self.single_interpreter.get_input_details()
        self.single_output = self.single_interpreter.get_output_details()

        # Load two-hand TFLite model
        self.two_interpreter = tf.lite.Interpreter(model_path=two_hand_model_path)
        self.two_interpreter.allocate_tensors()
        self.two_input = self.two_interpreter.get_input_details()
        self.two_output = self.two_interpreter.get_output_details()

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def normalize_landmarks(self, landmarks):
        base_x, base_y = landmarks[0].x, landmarks[0].y
        normalized = []
        for lm in landmarks:
            x = lm.x - base_x
            y = lm.y - base_y
            normalized.extend([x, y])
        return normalized

    def predict(self, input_vector, hand_count):
        input_array = np.array([input_vector], dtype=np.float32)
        if hand_count == 1:
            # single-hand uses first 42 values
            self.single_interpreter.set_tensor(self.single_input[0]['index'], input_array[:, :42])
            self.single_interpreter.invoke()
            result = self.single_interpreter.get_tensor(self.single_output[0]['index'])
        elif hand_count == 2:
            self.two_interpreter.set_tensor(self.two_input[0]['index'], input_array)
            self.two_interpreter.invoke()
            result = self.two_interpreter.get_tensor(self.two_output[0]['index'])
        else:
            return None
        return int(np.argmax(result))

    def run(self):
        cap = cv2.VideoCapture(0)
        prev_time = 0

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        ) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                prediction_text = "No hands detected"
                input_vector = []
                hand_count = 0

                if result.multi_hand_landmarks:
                    hand_count = len(result.multi_hand_landmarks)
                    all_landmarks = []

                    for hand_landmarks in result.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        normalized = self.normalize_landmarks(hand_landmarks.landmark)
                        all_landmarks.append(normalized)

                    if hand_count == 1:
                        input_vector = all_landmarks[0] + [0.0] * 42
                    elif hand_count == 2:
                        input_vector = all_landmarks[0] + all_landmarks[1]

                    if len(input_vector) == 84:
                        try:
                            label = self.predict(input_vector, hand_count)
                            prediction_text = f"Predicted Label: {label}"
                        except Exception as e:
                            prediction_text = f"Prediction Error: {str(e)}"

                # FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time

                # Display
                cv2.putText(frame, prediction_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'FPS: {int(fps)}', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.imshow("ISL Gesture Prediction", frame)

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break

        cap.release()
        cv2.destroyAllWindows()

# === ENTRY POINT ===
if __name__ == "__main__":
    # Here the constructor is called:
    classifier = DualHandGestureClassifier(
        single_hand_model_path='model/keypoint_classifier/keypoint_classifier_singlehand.tflite',
        two_hand_model_path='model/keypoint_classifier/keypoint_classifier_twohand.tflite'
    )
    # Then we start the webcam prediction loop
    classifier.run()
