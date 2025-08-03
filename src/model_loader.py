# src/model_loader.py

import os
import tensorflow as tf
import requests

MODEL_PATH = 'models/mnist_model.h5'
MODEL_URL = "https://drive.google.com/uc?export=download&id=1kJhLmabiVo8kBvEJpbyeejR034CAPM3i"

model = None

def download_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(MODEL_PATH):
        print("🔄 Downloading model from Google Drive...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded.")

def get_model():
    global model
    if model is None:
        download_model()
        print("📦 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def reload_model():
    global model
    download_model()
    print("♻️ Reloading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
