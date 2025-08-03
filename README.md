# 🧠 MNIST Digit Classifier — End-to-End ML Pipeline with Retraining

This project demonstrates a complete Machine Learning (ML) lifecycle using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The pipeline includes data acquisition, preprocessing, model training, evaluation, deployment via Flask API, and a Streamlit UI for interaction, visualization, and retraining.

---

## 📌 Project Overview

| Feature | Description |
|--------|-------------|
| **Dataset** | MNIST Handwritten Digit Dataset (60,000 training / 10,000 test) |
| **Model** | CNN built using TensorFlow/Keras |
| **UI** | Streamlit Web App |
| **API** | Flask-based REST API |
| **Retraining** | Trigger-based retraining on uploaded image ZIP files |
| **Deployment** | Ready for cloud deployment (e.g., GCP, AWS, Azure, etc.) |
| **Monitoring** | Locust load testing (optional step) |
| **Evaluation** | Accuracy, Precision, Recall, F1 Score, Confusion Matrix |

---

## 🔁 Key Functionalities

- ✅ **Predict** single digit from image upload (28x28 grayscale)
- 📈 **View visualizations**:
  - Class distribution
  - Confusion matrix
  - Average digit images
- 📦 **Upload ZIP** of images to retrain model on new data
- 🔁 **Trigger retraining** of the model using new uploaded data
- 🚀 **REST API** for external access to prediction and retraining
- 🧪 **Model Evaluation**: Accuracy, Precision, Recall, F1 Score

---

## 📁 Project Structure

