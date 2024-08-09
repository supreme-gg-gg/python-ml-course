import numpy as np
import math

# squared error cost function
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0.0

    for i in range(m):
        total_cost += (w * x[i] + b - y[i]) ** 2
    
    return total_cost / (2*m)

# gradient of the cost function
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dw_dj = 0.0
    db_dj = 0.0

    for i in range(m):
        f_wb = w*x[i] + b
        dw_dj += (f_wb - y[i]) * x[i]
        db_dj += f_wb - y[i]
    
    return dw_dj/m, db_dj/m

# Gradient Descent algorithm to minimize the cost function (w,b)
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = w_in
    b = b_in
    J_history = [] # history of cost function
    p_history = [] # history of parameters (plotting)

    for i in range(num_iters):
        dw, db = gradient_function(x, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db

        if i < 100000:
            J = cost_function(x, y, w, b)
            J_history.append(J)
            p_history.append((w, b))

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dw: 0.3e}, dj_db: {db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
            
    return w, b, J_history, p_history

# Sample training data and initial setup
x_training = np.array([1.0,2.0])
y_training = np.array([300.0, 500.0])

w_init = 0.0
b_init = 0.0
iterations = 10
learning_rate = 0.01

# You can choose to plot the cost function and the parameters
w_final, b_final, J_hist, p_hist = gradient_descent(x_training, y_training, w_init, b_init, learning_rate, iterations, compute_cost, compute_gradient)

# Predictions
def prediction(x, w, b):
    '''
    x input is a nd array from numpy with m examples
    '''
    m = x.shape[0] # basically len(x)
    f_wb = np.zeros(m) # initialize the output array
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

x_test = np.array([3.0, 4.0, 5.0])
predictions = prediction(x_test, w_final, b_final)
print(predictions)