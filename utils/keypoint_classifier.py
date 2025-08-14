#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier:
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Automatically determine number of classes from output shape
        self.num_classes = self.output_details[0]['shape'][-1]

    def __call__(self, landmark_list):
        """
        Accepts a flattened list of normalized landmarks (1 or 2 hands).
        Example shape: (42,) for 1 hand, (84,) for 2 hands (21 landmarks per hand × 2D).
        """

        if not isinstance(landmark_list, (list, np.ndarray)):
            raise ValueError("Input must be a list or NumPy array of landmarks")

        landmark_array = np.array([landmark_list], dtype=np.float32)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], landmark_array)

        # Run inference
        self.interpreter.invoke()

        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Result: index of the most probable class
        predicted_index = int(np.argmax(output))

        return predicted_index
