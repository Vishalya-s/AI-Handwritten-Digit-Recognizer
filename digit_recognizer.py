import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Load the MNIST handwritten digit dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

# Normalize pixel values

x_train = x_train / 255.0
x_test = x_test / 255.0

print("Minimum pixel value:", x_train.min())
print("Maximun pixel values:",x_test.max())

# Build the neutal network

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Show the model structure
model.summary()

# Compile the model
model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"]
              )
print("Model compiled successfully!")

# Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1
)

print("Training completed!")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make predictions
predictions = model.predict(x_test[:5], verbose=0)

# Get the predicted digit for each image
predicted_classes = predictions.argmax(axis=1)

print("Actual digits:", y_test[:5])
print("Predicted digits:", predicted_classes)

import matplotlib.pyplot as plt

# Display one handwritten digit 
plt.imshow(x_test[0], cmap="gray")
plt.title(f"Actual digit: {y_test[0]}")
plt.axis("off")
plt.show()

# Predict the displayed digit
prediction = model.predict(x_test[0:1], verbose=0)
predicted_digit = prediction.argmax(axis=1)[0]

print("Actual digit:", y_test[0])
print("Predicted digit:", predicted_digit)

# Save the trained model
model.save("digit_recognizer.keras")

print("Digit recognizer model saved successfully!")

# Load the saved model
loaded_model = tf.keras.models.load_model("digit_recognizer.keras")

print("Model loaded successfully!")

# Test the loaded model
loaded_prediction = loaded_model.predict(x_test[0:1], verbose=0)

loaded_digit = loaded_prediction.argmax(axis=1)[0]

print("Actual digit:", y_test[0])
print("Loaded model prediction:", loaded_digit)