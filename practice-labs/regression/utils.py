import numpy as np
import matplotlib.pyplot as plt
import math

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

def plot_data(X, y, pos_label = "y=1", neg_label = "y=0"):
    # Find Indices of Positive and Negative Examples
    positive = y == 1
    negative = y == 0

    # Plot Examples, X is a two-dimensional matrix
    # e.g. X[positive, 0] is the first feature of the positive examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label = pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label = neg_label)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b): 
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb = sigmoid(z) # the model prediction
        loss = -y[i]*np.log(f_wb) - (1-y[i])*np.log(1-f_wb)
        cost += loss
    cost /= m
    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_db = 0.0
    dj_dw = np.zeros_like(w)
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb - y[i]
        dj_db += err_i
        for j in range(n):
            dj_dw[j] += err_i * X[i][j]
    dj_db /= m
    dj_dw /= m
    return dj_dw, dj_db

# Recall the unit on feature engineering and polynomial regression
def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    J_history = []
    w_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w_in, b_in)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db
        # Save cost J at each iteration
        if i<100000:
            J_history.append(cost_function(X, y, w_in, b_in))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    
    return w_in, b_in, J_history, w_history