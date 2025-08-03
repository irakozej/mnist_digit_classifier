# src/model_loader.py

from tensorflow.keras.models import load_model

MODEL_PATH = 'models/mnist_model.h5'

# Global model variable
model = load_model(MODEL_PATH)

def get_model():
    global model
    return model

def reload_model():
    global model
    model = load_model(MODEL_PATH)
    return model
