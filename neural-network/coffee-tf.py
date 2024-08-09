import numpy as np
import tensorflow as tf
from utils import load_data

X, Y = load_data()
print(X.shape, Y.shape)

# Noramlize the data since back-propagation will be more efficient this way
norm_l = tf.keras.layers.Normalization(axis = -1) # not a layer of the model
norm_l.adapt(X) # learns mean, variance of the data set and saves the values internally
Xn = norm_l(X)

''' Make sure to also normalise any future data that utilizes the learned model '''

# Tile/copy our data to increase the training set size and reduce the number of training epochs
Xt = np.tile(Xn, (1000,1))
Yt = np.tile(Y, (1000,1))
print(Xt.shape, Yt.shape)

tf.random.set_seed(1234)

'''
    Input(shape()) specifies the expected shape of the input 
    for TYensorflow to size the weights and bias. This can be omitted
    and Tensorflow will size the parameters when the input is specified
    in the model.fit statement
'''

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(2,)), # This is the input layer
    tf.keras.layers.Dense(3, activation="sigmoid", name="layer1"),
    tf.keras.layers.Dense(1, activation="sigmoid", name="layer2")
])
model.summary() # there are 13 parameters in total, think about where they are from!

# model.compile defines a loss function and specifies a compile optimization
# model.fit runs gradient descent and fits the weights to the data
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
)
model.fit(
    Xt, Yt, 
    epochs=10,
)

# After fitting, the weights have been updated
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

# Now let's make predictions using the trained model
X_test = np.array([
    [200, 13.9],
    [200, 17]
])
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("Predictions:\n", predictions)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
# or simply: yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")