'''
Supose you are the CEO of a restaurant franchise and 
are considering different cities to expand. 
You are given X-train as population of cities 
and y_train as profits of restaurant in that city
'''

import numpy as np
import matplotlib.pyplot as plt
import copy

def load_data():
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y

x_train, y_train = load_data()
print("First five elements of x_train are:\n", x_train[:5]) 
print("First five elements of y_train are:\n", y_train[:5])  

def compute_cost(x,y,w,b):
    m = x.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = x[i] * w + b
        total_cost += (f_wb - y[i]) ** 2

    return total_cost / (2*m)

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = x[i] * w + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    return dj_dw/m, dj_db/m

def gradient_descent(x, y, w_in, b_in, compute_gradient, alpha, num_iters):
    m = len(x)
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
    return w, b

initial_w = 0.
initial_b = 0.
iterations = 1500
alpha = 0.01

w, b = gradient_descent(x_train, y_train, initial_w, initial_b, compute_gradient, alpha, iterations)
print("w, b found by gradient descent:", w, b)

# Test the prediction using training sample
m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w * x_train[i] + b

# Let's visualise the data first
plt.scatter(x_train, y_train, marker = 'x', c='r')
plt.plot(x_train, predicted, c = "b")
plt.title("Profit vs Population per city")
plt.ylabel("Profit in $10,000")
plt.xlabel("Population in 10,000")
plt.show

print('For population = 35,000, we predict a profit of $%.2f' % ((3.5 * w + b) * 10000))
print('For population = 70,000, we predict a profit of $%.2f' % ((7.0 * w + b) * 10000))