import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *

# First load and visualise the data, as usual

X_train, y_train = load_data('data/ex2data1.txt')

# We have 100 training examples and 2 features

# y = 1 is the positive class (admitted)
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not Admitted")

plt.ylabel('Exam 2 score')
plt.xlabel('Exam 1 score')
plt.legend(loc="upper right")
plt.show()

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2).reshape(-1, 1) - 0.5)
initial_b = -8
iterations = 10000
alpha = 0.001
w, b, J_history, w_history = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, iterations, compute_cost, compute_gradient)

# For simplicity this only plots the decision boundary for 2D data
def plot_decision_boundary(w, b, X, y):

    plot_data(X[:, 0:2], y)
    
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        
        plt.plot(plot_x, plot_y, c="b")
    else: 
        raise ValueError("Can only plot 2D data")
    
plot_decision_boundary(w, b, X_train, y_train)

def predict(X, w, b):
    m, n = X.shape
    y_pred = np.zeros(m)
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i], w) + b)
        y_pred[i] = 1 if f_wb >= 0.5 else 0
    return y_pred

np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5
tmp_y = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_y.shape}, value {tmp_y}')

# Test the model on the training data
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))
