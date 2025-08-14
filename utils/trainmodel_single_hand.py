# ==== Single-Hand Gesture Model (42 keypoints) ====
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ==== CONFIG ====
NUM_KEYPOINTS = 42
NUM_CLASSES = 30
dataset_path = 'model/keypoint_classifier/keypoint_singlehand.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier_singlehand.hdf5'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier_singlehand.tflite'

# ==== LOAD DATA ====
X = np.loadtxt(dataset_path, delimiter=',', dtype='float32', usecols=list(range(1, NUM_KEYPOINTS + 1)))
y = np.loadtxt(dataset_path, delimiter=',', dtype='int32', usecols=(0,))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

# ==== BUILD MODEL ====
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((NUM_KEYPOINTS,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_weights_only=False, verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=500, batch_size=128, validation_data=(X_test, y_test),
                    callbacks=[cp_callback, es_callback])

# ==== EXPORT ====
model.save(model_save_path, include_optimizer=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(tflite_save_path, 'wb') as f:
    f.write(tflite_model)
print("✅ Saved single-hand model.")
