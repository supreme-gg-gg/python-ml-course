import numpy as np
# import matplotlib.pyplot as plt
import math

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

'''
Note that this would also work with vectors and matrices. 
The function will return a vector or a matrix of the same size as z, 
where each element is the sigmoid of the corresponding element of z.
'''

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

# If you want to use regularization set lambda_ to a value > 0 (e.g. 1)
def compute_cost_logistic(X, y, w, b, lambda_ = 0.0):
    m,n = X.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i) # see notes
    cost /= m

    # The following code is for regularization
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j] ** 2
    reg_cost *= (lambda_/(2*m))
    cost += reg_cost

    return cost

def compute_gradient_logistic(X, y, w, b, lambda_ = 0.0):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i][j]
        dj_db += err_i
    dj_db /= m
    dj_dw /= m

    # The following code is for regularization
    for j in range(n):
        dj_dw[j] += w[j] * (lambda_/m)
    
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_dw , dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(compute_cost_logistic(X, y, w, b))
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i}, cost: {J_history[-1]}")
    return w, b, J_history

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 