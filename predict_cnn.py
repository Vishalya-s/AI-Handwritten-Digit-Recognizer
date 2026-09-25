import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load CNN model
model = tf.keras.models.load_model("cnn_digit_recognizer.keras")

# Load handwritten digit
image = Image.open("handwritten_5_mnist.png").convert("L")

# Convert to NumPy array
image = np.array(image)

# Normalize
image = image / 255.0

# Show the exact image given to the CNN
plt.imshow(image, cmap="gray")
plt.title("Image given to CNN")
plt.axis("off")
plt.show()

# Prepare image for CNN
input_image = image.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(input_image, verbose=0)

predicted_digit = prediction.argmax(axis=1)[0]

print("CNN predicted digit:", predicted_digit)