import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers

# Neuron with linear activation 
X_train = np.array([[1.0], [2.0]], dtype=np.float32)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)

linear_layer = layers.Dense(units=1, activation ='linear')

# Let's randomly assign some weights and bias
set_w = np.array([[200]])
set_b = np.array([100])
# set_weights take a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

a1 = linear_layer(X_train[0].reshape(1,1)) # Input to the layer must be 2D
print(a1) # Output: tf.Tensor([[300.]], shape=(1, 1), dtype=float32)
alin = np.dot(set_w, X_train[0].reshape(1,1)) + set_b
print(alin) # Output: [[300.0]], same value as a1

# We can now use the linear layer to make predictions on our training data
prediction_tf = linear_layer(X_train)
prediction_np = np.dot(X_train, set_w) + set_b

# Neuron with logistic activation
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)

model = keras.Sequential(
    [
        layers.Dense(units=1, activation='sigmoid', input_dim=1, name='L1')
    ]
)
model.summary()
logistic_layer = model.get_layer('L1')
set_w = np.array([[2]])
set_b = np.array([-4.5])
logistic_layer.set_weights([set_w, set_b])
print(logistic_layer.get_weights())

a2 = model.predict(X_train[0].reshape(1,1))
print(a1) # [[0.01]]
alog = keras.activations.sigmoid(np.dot(set_w, X_train[0].reshape(1,1)) + set_b)
print(alog) # [[0.01]], same value as a2