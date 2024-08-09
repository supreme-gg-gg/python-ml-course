import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

# Softmax function takes in a vector of logits and returns a vector of probabilities
def softmax(z):
    ez = np.exp(z) # element-wise exponentiation
    sm = ez/np.sum(ez)
    return sm

# Generate a dataset with m samples and 4 output classes, 2 input features
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=1.0, random_state=30)
print(f"unique classes {np.unique(y_train)}")
print(f"class representation {y_train[:10]}")
print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")

# Use a two layer network: x -> 2 ReLU -> 4 linear -> softmax -> probabilities

tf.random.set_seed(1234) # applied to achieve consistent results
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='relu', name="L1"),
    tf.keras.layers.Dense(4, activation='linear', name="L2")
    
])

'''
    tf.keras.layers.Dense(4, activation='softmax', name="L3")
    This is not recommended because it is numerically unstable
'''

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

''' 
    From_logits informs the loss function that the softmax operation 
    should be included in the loss calculation 
'''

model.fit(X_train, y_train, epochs=200)

# For this model the outputs are not probabilities so 
# they must be first sent through a softmax when predicting probability

prediction = model.predict(X_train)
print(f"two examples of predictions: {prediction[:2]}")
sm_prediction = tf.nn.softmax(prediction)
print(f"two examples of softmax predictions: {sm_prediction[:2]}")

# However, if you just want to select the most likely category softmax is not required
# You can just select the category with the highest raw output

for i in range(5):
    print(f"{prediction[i]} category: {np.argmax(prediction[i])}")