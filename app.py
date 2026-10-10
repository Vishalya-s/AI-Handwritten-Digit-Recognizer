import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page settings
st.set_page_config(
    page_title="AI Digit Recognizer",
    page_icon="🔢"
)

# Title
st.title("🔢 AI Handwritten Digit Recognizer")

st.markdown(
    """
    ### 🧠 CNN-Powered Digit Recognition
    Upload a handwritten digit and let the trained Convolutional Neural Network
    recognize it.

    *Model:* CNN trained on the MNIST dataset  
    *Test Accuracy:* 98.89%
    """
)

# Load CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_recognizer.keras")

model = load_model()

# Sidebar
st.sidebar.title("📌 About the Project")

st.sidebar.write(
    """
    This application uses a Convolutional Neural Network (CNN)
    trained on the MNIST handwritten digit dataset.

    The model can recognize digits from 0 to 9.
    """
)

st.sidebar.metric("Model Test Accuracy", "98.89%")
st.sidebar.info("Built with TensorFlow + Streamlit")

if st.sidebar.button("🔀 Reset"):
    st.rerun()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Display uploaded image
    st.image(
        uploaded_file,
        caption="Uploaded image",
        width=250
    )

    # Prediction button
    if st.button("🔍 Predict Digit"):

        # Open image
        image = Image.open(uploaded_file).convert("L")

        # Resize to MNIST size
        image = image.resize((28, 28))

        # Convert to NumPy array
        image = np.array(image)

        # Normalize
        image = image / 255.0

        # Prepare image for CNN
        image = image.reshape(1, 28, 28, 1)

        # Make prediction
        prediction = model.predict(image, verbose=0)

        # Get predicted digit
        predicted_digit = prediction.argmax(axis=1)[0]

        # Get confidence
        confidence = prediction[0][predicted_digit] * 100

        # Display result
        st.success(f"Predicted digit: {predicted_digit}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Display probabilities
        st.subheader("Prediction probabilities")

        probabilities = prediction[0] * 100

        for digit, probability in enumerate(probabilities):
            st.write(f"Digit {digit}: {probability:.2f}%")
            st.progress(float(probability / 100))

st.markdown("---")
st.caption("AI Handwritten Digit Recognizer | Built with TensorFlow and Streamlit")
