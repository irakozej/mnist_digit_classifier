# src/model_loader.py

import os
import tensorflow as tf
import requests

# Suppress TensorFlow info & warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"  # Reduce memory usage

MODEL_PATH = 'models/mnist_model.h5'
MODEL_URL = "https://drive.google.com/uc?export=download&id=1kJhLmabiVo8kBvEJpbyeejR034CAPM3i"

# Global cache
model = None

def download_model():
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("🔄 Downloading model from Google Drive...")
        try:
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
                print("✅ Model downloaded.")
            else:
                print(f"❌ Failed to download model. Status code: {response.status_code}")
        except Exception as e:
            print(f"❌ Exception during model download: {e}")

def get_model():
    global model
    if model is None:
        download_model()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH} after attempted download.")
        print("📦 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def reload_model():
    global model
    print("♻️ Reloading model...")
    download_model()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH} after attempted download.")
    model = tf.keras.models.load_model(MODEL_PATH)
