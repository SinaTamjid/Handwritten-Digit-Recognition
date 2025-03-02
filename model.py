#importing packages
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.api.datasets import mnist
from keras.api.layers import Dense, Flatten, InputLayer  # Include InputLayer here
from keras.api.models import Sequential
from keras.api.utils import to_categorical
import matplotlib.pyplot as plt

#loading dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Expand the dimensions to include the batch size and channels
X_train = np.expand_dims(X_train, axis=-1)  # Shape: (60000, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=-1)    # Shape: (10000, 28, 28, 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Building the model
model = Sequential([
    InputLayer(input_shape=(28, 28, 1)),  # Use InputLayer to specify the input shape
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the data
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluation
evaluate = model.evaluate(X_test, y_test)
print(f"The evaluation for this model is: {evaluate}")

tf.saved_model.save(model,r'py\NLP\hand write\Handwritten-Digit-Recognition\model')
# Visualize some predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    # Prediction for some test images
    prediction = model.predict(np.expand_dims(X_test[i], axis=0))
    plt.title(f"Prediction for this image is : {np.argmax(prediction)}")
    plt.axis('off')
plt.show()

