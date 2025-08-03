# src/retrain.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "models/mnist_model.h5"
HISTORY_PATH = "models/history.npy"

# Helper to build a CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Main retraining function
def retrain_model(custom_data_path=None):
    model = build_model()

    if custom_data_path:
        # Retrain on uploaded data
        datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            custom_data_path,
            target_size=(28, 28),
            color_mode="grayscale",
            class_mode="categorical",
            subset='training',
            batch_size=32
        )

        val_gen = datagen.flow_from_directory(
            custom_data_path,
            target_size=(28, 28),
            color_mode="grayscale",
            class_mode="categorical",
            subset='validation',
            batch_size=32
        )

        history = model.fit(train_gen, validation_data=val_gen, epochs=5)

    else:
        # Retrain on original MNIST
        (x_train, y_train), (x_val, y_val) = mnist.load_data()
        x_train = x_train / 255.0
        x_val = x_val / 255.0

        x_train = np.expand_dims(x_train, -1)  # (num, 28, 28, 1)
        x_val = np.expand_dims(x_val, -1)

        y_train = to_categorical(y_train, 10)
        y_val = to_categorical(y_val, 10)

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5)

    model.save(MODEL_PATH)
    np.save(HISTORY_PATH, history.history)
    print("✅ Model retrained and saved.")
