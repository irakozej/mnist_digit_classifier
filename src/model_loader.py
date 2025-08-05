# src/model_loader.py

import os
import tensorflow as tf
import requests

# ✅ Suppress TensorFlow logs (only errors will show)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 3 = errors only

# ✅ Limit CPU threads to reduce memory usage
os.environ["OMP_NUM_THREADS"] = "1"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Model storage path and download link
MODEL_PATH = 'models/mnist_model.h5'
MODEL_URL = "https://drive.google.com/uc?export=download&id=1kJhLmabiVo8kBvEJpbyeejR034CAPM3i"

# Cached model object
model = None

def download_model():
    """Download model from Google Drive if not present."""
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("🔄 Downloading model from Google Drive...")
        try:
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
                print("✅ Model downloaded successfully.")
            else:
                print(f"❌ Failed to download model. HTTP status code: {response.status_code}")
        except Exception as e:
            print(f"❌ Exception while downloading model: {e}")

def get_model():
    """Lazy-load and return the model."""
    global model
    if model is None:
        download_model()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH} after attempted download.")
        print("📦 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def reload_model():
    """Force reload the model (e.g., after retraining)."""
    global model
    print("♻️ Reloading model...")
    download_model()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH} after attempted download.")
    model = tf.keras.models.load_model(MODEL_PATH)
