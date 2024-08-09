import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *

X_train, y_train = load_data('data/ex2data2.txt')
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")
plt.ylabel('Microchip Test 2')
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

# We can see that our data cannot be separated by a straight line

'''
One way to fit the data better is to create more features form each data point
We will map the features into all polynomial terms of x1 and x2 up to the sixth power
A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and 
will appear nonlinear when drawn in our 2D plot.
'''
print("Original shape of data:", X_train.shape)
X_train = map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", X_train.shape)

# We will implmement regularization to help combat the overfitting problem

def compute_cost_reg(X, y, w, b, lambda_ = 1):
    m, n = X.shape
    cost = compute_cost(X, y, w, b)
    
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j] ** 2
    # another way: reg_cost = sum(np.square(w))
    reg_cost *= (lambda_ / (2*m))

    return cost + reg_cost

np.random.seed(1)
initial_w = np.random.rand(X_train.shape[1]) - 0.5
initial_b = 0.5
cost = compute_cost_reg(X_train, y_train, initial_w, initial_b)
print(f"Initial cost: {cost:.2f}")

def compute_gradient_reg(X, y, w, b, lambda_ = 1):
    m, n = X.shape
    dj_dw, dj_db = compute_gradient(X, y, w, b)

    for j in range(n):
        dj_dw[j] += lambda_ * w[j] / m
    
    return dj_dw, dj_db

dj_dw, dj_db = compute_gradient_reg(X_train, y_train, initial_w, initial_b)
print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

w, b, J_history, _ = gradient_descent(X_train, y_train, initial_w, initial_b, 0.01, 10000, compute_cost_reg, compute_gradient_reg)

# This ends here for now but you can plot decision boundary and test the accuracy just as in university.py