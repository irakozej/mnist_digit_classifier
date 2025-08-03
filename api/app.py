# api/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import zipfile
import shutil

from src.retrain import retrain_model
from src.model_loader import get_model, reload_model

app = Flask(__name__)

MODEL_PATH = "models/mnist_model.h5"
UPLOAD_DIR = "data/new_upload"

# ✅ Automatically retrain model if it's missing (important for Render)
if not os.path.exists(MODEL_PATH):
    print("⚠️ Model not found. Starting retraining...")
    os.makedirs("models", exist_ok=True)
    retrain_model()

# Helper: preprocess uploaded image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=(0, -1))  # Shape: (1, 28, 28, 1)
    return image

@app.route('/')
def home():
    return "MNIST Digit Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = preprocess_image(file.read())
    prediction = get_model().predict(image)
    predicted_class = int(np.argmax(prediction))
    confidence_scores = prediction[0].tolist()

    return jsonify({
        'prediction': predicted_class,
        'confidence': confidence_scores
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        retrain_model()
        reload_model()
        return jsonify({'message': 'Model retrained and reloaded successfully!'})
    except Exception as e:
        print("❌ Error during /retrain:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/upload_and_retrain', methods=['POST'])
def upload_and_retrain():
    if 'zipfile' not in request.files:
        return jsonify({'error': 'No zipfile uploaded'}), 400

    zip_file = request.files['zipfile']

    # Clean previous
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_DIR)
    except Exception as e:
        print("❌ Error extracting ZIP:", e)
        return jsonify({'error': f'Failed to extract ZIP file: {e}'}), 500

    try:
        retrain_model(custom_data_path=UPLOAD_DIR)
        reload_model()
        return jsonify({'message': 'Retrained with uploaded data'}), 200
    except Exception as e:
        print("❌ Error during retraining:", e)
        return jsonify({'error': f'Retraining failed: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
