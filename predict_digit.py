import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("digit_recognizer.keras")

print("Model loaded successfully!")

# Load your handwritten digit
image = Image.open("handwritten_5_mnist.png").convert("L")

# Convert image to NumPy array
image = np.array(image)

# Normalize pixel values
image = image / 255.0

# Make prediction
prediction = model.predict(image.reshape(1, 28, 28), verbose=0)

predicted_digit = prediction.argmax(axis=1)[0]

print("Predicted digit:", predicted_digit)

# Display the image
plt.imshow(image, cmap="gray")
plt.title(f"Predicted digit: {predicted_digit}")
plt.axis("off")
plt.show()