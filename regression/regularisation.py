import numpy as np
import copy
import math
import matplotlib.pyplot as plt

def load_house_data():
    data = np.loadtxt("./houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

# If you want to use regularization set lambda_ to a value > 0 (e.g. 1)
def compute_cost(X, y, w, b, lambda_=0.0):
    '''
    Args: 
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): weights
        b (scalar): bias

    Returns: 
        cost (scalar): cost function
    '''

    m = X.shape[0] # X.shape would return (m,n)
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(w, X[i]) + b
        cost += (f_wb_i - y[i]) ** 2
    cost /= (2*m)

    # The following code is for regularization
    n = len(w)
    reg_cost = 0
    for j in range(n):
        reg_cost += w[j] ** 2
    reg_cost = (lambda_/(2*m))*reg_cost
    cost += reg_cost

    return cost

def compute_gradient(X, y, w ,b, lambda_=0.0):
    m, n = X.shape # (number of examples, number of features)
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i][j]

        dj_db += err
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    # The following code is for regularization
    for j in range(n):
        dj_dw[j] += w[j] * (lambda_/m)

    return dj_db, dj_dw

# We skipped JHistory here so refer to the last file for that
def gradient_descent(X, y, w_in, b_in, cost_function, compute_gradient, alpha, num_iters = 1000):
    
    hist = {
        "cost": [],
        "params": [],
        "grads": [],
        "iter": []
    }
    
    w = copy.deepcopy(w_in)
    b = b_in
    save_interval = np.ceil(num_iters / 10000)

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i == 0 or i % save_interval == 0:      # prevent resource exhaustion 
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append((w, b))
            hist["grads"].append((dj_dw, dj_db))
            hist["iter"].append(i)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            #print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
            cst = cost_function(X, y, w, b)
            print(f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")

    return w, b, hist 


def run_gradient_descent(X, y, iter = 1000, alpha = 1.0e-1) :
    
    initial_w = np.zeros(X.shape[1])
    initial_b = 0.0
    w_out, b_out, hist_out = gradient_descent(X, y, initial_w, initial_b, compute_cost, compute_gradient, alpha, iter)
    print(f"w, b: {w_out}, {b_out}")

    return (w_out, b_out, hist_out)

# The theory behind refer to lecture notes
# With scaling, we can run gradient descent with a larger alpha
def zscore_normalize(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    return (X_norm, mu, sigma)

# This function (not important) is used to plot the cost function and the first parameter (i.e. learning curve) to see how the cost function changes with the number of iterations & learning rate
def plot_cost_i_w(X,y,hist):
    ws = np.array([ p[0] for p in hist["params"]])
    rng = max(abs(ws[:,0].min()),abs(ws[:,0].max()))
    wr = np.linspace(-rng+0.27,rng+0.27,20)
    cst = [compute_cost(X,y,np.array([wr[i],-32, -67, -1.46]), 221) for i in range(len(wr))]

    fig,ax = plt.subplots(1,2,figsize=(12,3))
    ax[0].plot(hist["iter"], (hist["cost"]));  ax[0].set_title("Cost vs Iteration")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("Cost")
    ax[1].plot(wr, cst); ax[1].set_title("Cost vs w[0]")
    ax[1].set_xlabel("w[0]"); ax[1].set_ylabel("Cost")
    ax[1].plot(ws[:,0],hist["cost"])
    plt.show()

# Load data and run gradient descent
X_train, y_train = load_house_data()
X_norm, X_mu, X_sigma = zscore_normalize(X_train)
w_norm, b_norm = run_gradient_descent(X_norm, y_train)

# Predictions (first noramlize input)
x_input = np.array([1650, 3, 880, 7])
x_input_norm = (x_input - X_mu) / X_sigma
y_pred = np.dot(x_input_norm, w_norm) + b_norm

# An example of feature engineering could be a polynomial fit
x = np.arange(0, 20, 1)
y = x**2
X = np.c_[x, x**2, x**3] # Added engineered features

# we can also add in feature scaling
X = zscore_normalize(X)

w, b, hist = run_gradient_descent(X, y)
# if you plot this it will be a much better fit than just a straight line