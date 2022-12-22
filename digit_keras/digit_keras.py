# import relevant libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from tensorflow import keras

# Download the data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
X_train = X_train.astype(np.float32)/255.0
X_test = X_test.astype(np.float32)/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_shape = (28, 28, 1)
num_classes = y_train.shape[1]

# Initialise the model
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), use_bias=True, input_shape=input_shape,
                        kernel_initializer="glorot_uniform", activation='relu', padding='same', bias_initializer="zeros"),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
                        kernel_initializer="glorot_uniform", activation='relu', padding='same', bias_initializer="zeros"),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(784, activation='relu',
                       kernel_initializer=keras.initializers.HeNormal()),
    keras.layers.Dense(256, activation='relu',
                       kernel_initializer=keras.initializers.HeNormal()),
    keras.layers.Dense(64, activation='relu',
                       kernel_initializer=keras.initializers.HeNormal()),
    keras.layers.Dense(num_classes, activation='linear')
])

# Compile and Train the model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.3,
          batch_size=256, epochs=20, verbose=1)

# Load the saved model
model = keras.models.load_model('digit_keras.h5')

# Predict
y_pred = model.predict(X_test)


# Evaluating the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Loss: {scores[0]}")
print(f"Accuracy on test set: {scores[1]*100}%")

# Save the model
model.save('digit_keras.h5')
