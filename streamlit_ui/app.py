# streamlit_ui/app.py

import streamlit as st
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

API_URL = "http://127.0.0.1:5000"

st.title("MNIST Digit Classifier")
st.write("Upload a handwritten digit image (28x28 grayscale) to predict the number.")

# Section: Predict single image
st.header("📸 Predict a Single Image")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=150)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/predict", files={"file": uploaded_file})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Predicted Digit: **{result['prediction']}**")

            # Show confidence scores
            confidence = result.get('confidence')
            if confidence:
                st.subheader("🔍 Prediction Confidence")
                fig, ax = plt.subplots()
                ax.bar(range(10), confidence, color='skyblue')
                ax.set_xlabel("Digit Class")
                ax.set_ylabel("Confidence")
                ax.set_xticks(range(10))
                st.pyplot(fig)
        else:
            st.error("❌ Prediction failed. Please try again.")

# Divider
st.markdown("---")

# Section: Retrain model with original MNIST
st.header("🔁 Retrain the Model")
st.write("Click the button below to retrain the model using the original MNIST dataset.")

if st.button("Retrain Model"):
    with st.spinner("Retraining model..."):
        response = requests.post(f"{API_URL}/retrain")
        if response.status_code == 200:
            st.success("✅ Model retrained successfully!")
        else:
            st.error("❌ Error during retraining.")

# Divider
st.markdown("---")

# Section: Upload New ZIP & Retrain
st.header("Upload New Images & Retrain")

uploaded_zip = st.file_uploader("Upload ZIP file of new digit data", type=["zip"])

if uploaded_zip and st.button("Retrain with Uploaded Data"):
    files = {"zipfile": uploaded_zip.getvalue()}
    with st.spinner("Uploading and retraining..."):
        response = requests.post(f"{API_URL}/retrain", files=files)  # ✅ fixed line
        if response.status_code == 200:
            st.success("✅ Model retrained with new uploaded data!")
        else:
            st.error("❌ Retraining failed. Check ZIP structure.")

# Divider
st.markdown("---")

# Section: Visualize training performance
st.header("📈 Model Training Performance")

if st.button("Show Training Curves"):
    try:
        history = np.load("models/history.npy", allow_pickle=True).item()
        st.subheader("Training Accuracy & Loss")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].plot(history['accuracy'], label="Train Acc")
        ax[0].plot(history['val_accuracy'], label="Val Acc")
        ax[0].legend()
        ax[0].set_title("Accuracy")

        ax[1].plot(history['loss'], label="Train Loss")
        ax[1].plot(history['val_loss'], label="Val Loss")
        ax[1].legend()
        ax[1].set_title("Loss")

        st.pyplot(fig)
    except:
        st.warning("⚠️ Training history not found. Retrain the model first.")

# Footer
st.markdown("---")
st.caption("Created for Machine Learning Pipeline Assignment")
