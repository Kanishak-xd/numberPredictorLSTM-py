import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Sequence of numbers
data = np.array([[i, i+1, i+2] for i in range(10)])
target = np.array([i+3 for i in range(10)])

# Reshape for LSTM (samples, time steps, features)
data = data.reshape((data.shape[0], data.shape[1], 1))

# Define model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(3, 1)),
    Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(data, target, epochs=200, verbose=0)

# Make a prediction
test_input = np.array([[10, 11, 12]]).reshape((1, 3, 1))
prediction = model.predict(test_input)
print("Predicted next number:", prediction[0][0])
