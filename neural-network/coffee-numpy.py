import numpy as np
from utils import load_data
# import tensorflow as tf

X, Y = load_data()
print(X.shape, Y.shape)

'''
norm_l = tf.keras.layers.Normalization(axis = -1)
norm_l.adapt(X)
Xn = norm_l(X)
'''

def z_score_normalize(X):
    Xn = np.zeros_like(X)
    for i in range(X.shape[1]):
        std = np.std(X[:, i])
        if std == 0:
            Xn[:, i] = 0
        else: 
            Xn[:, i] = (X[:, i] - np.mean(X[:, i])) / std
    return Xn

Xn = z_score_normalize(X)

print(f"Max, Min Temp: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}")
print(f"Max, Min Dur: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}")

def dense(a_in, W, b, g):
    """
    Computes dense layer
    This is not a vectorized implementation
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1] # Number of neurons that layer has
    a_out = np.zeros(units)
    for j in range(units):
        # For each neuron, there are n weights which are the rows of W
        w = W[:, j] # Extract all rows of the j-th column (j-th neuron)
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out

def sigmoid(z):
    return 1/(1 + np.exp(-z))

# Create a two layer neural network, x is input set
def sequential(x, W1, b1, W2, b2):
    a1 = dense(x, W1, b1, sigmoid)
    a2 = dense(a1, W2, b2, sigmoid)
    return a2

# For now we will not train the network but use the weights and biases from Tensorflow
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], 
                    [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], 
                    [-27.59], 
                    [-32.56]] )
b2_tmp = np.array( [15.41] )

# Making predictions
def predict(X, W1, b1, W2, b2):
    m = X.shape[0] # number of samples
    p = np.zeros((m,1))
    for i in range(m):
        p[i, 0] = sequential(X[i], W1, b1, W2, b2)
    return p

X_test = np.array([
    [200, 13.9],
    [200, 17]
])
# X_testn = norm_l(X_test)
X_testn = z_score_normalize(X_test)
predictions = predict(X_testn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else: 
        yhat[i] = 0
print(f"Predictions: {yhat} (1: Good, 0: Bad)")

# Vectorized implementation

def dense_vectorized(A_in, W, B, g):
    """
    A_in (ndarray (m, n)) : Data, m examples, n features
    A_out (ndarray (m, j))  : j units
    """
    A_out = g(np.matmul(A_in, W) + B)
    return A_out

def sequential_vectorized(X, W1, B1, W2, B2):
    a1 = dense_vectorized(X, W1, B1, sigmoid)
    a2 = dense_vectorized(a1, W2, B2, sigmoid)
    return a2

result = sequential_vectorized(X_testn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
predictions2 = result.flatten()
yhat2 = np.where(predictions2 >= 0.5, 1, 0)
print(f"Predictions: {yhat2} (1: Good, 0: Bad)")