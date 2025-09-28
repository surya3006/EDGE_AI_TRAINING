import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore
import matplotlib.pyplot as plt
import logging
import datetime
import io
import sys
import os


# Create a log file with timestamp in its name
log_filename = f"mnist_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,  # Log INFO and above (use DEBUG for even more details)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Also show logs in the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logging.info("🟢 MNIST Training Script Started")


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the images (28x28 -> 784)
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build a basic Neural Network model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Predict a few test images
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {np.argmax(y_test[i])}")
    plt.show()

# save the mode
model.save("my_model.keras")
model_size = os.path.getsize("my_model.keras")

print(f"Model size on disk: {model_size / 1024:.2f} KB")


total_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
float_size = 4  # 32-bit float = 4 bytes
model_memory = total_params * float_size
# Create a string buffer
model_summary = io.StringIO()

# Write model summary to the buffer instead of printing
model.summary(print_fn=lambda x: model_summary.write(x + '\n'))

# Log the model summary
logging.info("📄 Model Summary:\n" + model_summary.getvalue())

# Close the buffer
model_summary.close()
print(f"Estimated memory for model weights: {model_memory / 1024:.2f} KB")
