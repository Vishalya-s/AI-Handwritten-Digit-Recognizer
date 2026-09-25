# 🔢 AI Handwritten Digit Recognizer

A deep learning application that recognizes handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## 🧠 Project Overview

This project uses TensorFlow and a CNN to classify handwritten digit images.

The project includes:
- MNIST dataset training
- CNN-based image classification
- Model evaluation
- Custom handwritten digit prediction
- Streamlit web application
- Prediction confidence and probability display

## 🚀 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- Streamlit
- MNIST Dataset

## 📊 Model Performance

*Test Accuracy: 98.89%*

The CNN was trained for 5 epochs on the MNIST dataset.

## 🏗️ CNN Architecture

```text
Input Image (28 × 28 × 1)
        ↓
Conv2D (32 filters)
        ↓
MaxPooling2D
        ↓
Conv2D (64 filters)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128 neurons)
        ↓
Dense (10 neurons)
        ↓
Predicted Digit (0–9)

💻 Streamlit Application

The application allows users to:

1. Upload a handwritten digit image
2. Click Predict Digit
3. View the predicted digit
4. View prediction confidence
5. View probabilities for all digits from 0 to 9

📁 Project Structure

digit-recognizer/
│
├── app.py
├── digit_recognizer.py
├── cnn_digit_recognizer.py
├── predict_digit.py
├── predict_cnn.py
│
├── digit_recognizer.keras
├── cnn_digit_recognizer.keras
│
├── handwritten_5_mnist.png
│
└── README.md

▶️ How to Run

Install the required packages:

pip install tensorflow numpy matplotlib pillow streamlit

Run the Streamlit application:

python -m streamlit run app.py

The application will open in your browser.

🎯 Learning Outcomes

Through this project, I learned:

• TensorFlow fundamentals
• Neural networks
• CNNs
• Image preprocessing
• Model training and evaluation
• Model saving and loading
• Prediction probabilities
• Streamlit application development
• Deploying a machine learning model into an interactive application

👩‍💻 Author

Vishalya S