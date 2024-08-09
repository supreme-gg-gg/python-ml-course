import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

'''
Scikit-learn has a gradient descent regression model sklearn.linear_model.SGDRegressor
Like your previous implementation of gradient descent, this model performs best with normalized inputs.
StandardScalar will perform z-score normalization on the data.  
'''

def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

X_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

# Note, the parameters are associated with the normalized input data
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b: {b_norm}")

# Predictions: use the predict routine OR compute using w, b (same output)
y_pred_sgd = sgdr.predict(X_norm)
y_pred = np.dot(X_norm, w_norm) + b_norm
print(f"Prediction on training set:\n{y_pred[:4]}" )

# Plot the prediction versus the target values
# It's not hard to figure out how most of this works with little experience with plt
fig, ax = plt.subplots(1, 4, figsize = (12, 3), sharey = True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label = "target")
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:, i], y_pred, color = "orange", label = "predict")  
ax[0].set_ylabel("price"); ax[0].legend();
fig.suptitle("Predictions vs. Target using z-score noramlized model")
plt.show()

# Implement linear regression using a close form solution based on the normal equation
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()

X_train = np.array([1.0, 2.0])
y_train = np.array([300, 500])

linear_model.fit(X_train.reshape(-1, 1), y_train)

# w, b are referred to as "coefficients" and "intercept" in scikit-learn
b = linear_model.intercept_
w = linear_model.coef_
print(f"model parameters: w: {w}, b: {b}")
y_pred = linear_model.predict(X_train.reshape(-1, 1))
print(f"Prediction on training set:\n{y_pred}")

X_test = np.array([1200])
print(f"Prediction for 1200 sqft: ${linear_model.predict(X_test)[0]}")

# The closed-form solution does not require normalization (works well only on small data sets)
# You can also involve multiple features but it's just similar to above

# Logistic regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])
lr_model.fit(X_train2, y_train2)

y_pred = lr_model.predict(X_train2)
print(f"Predictions: {y_pred}")

print("Accuracy on training set:", lr_model.score(X_train2, y_train2))