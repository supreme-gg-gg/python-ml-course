import torch
import torch.nn as nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

X = torch.arange(start=0, end=1, step=0.02).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
x_train, y_train = X[:train_split], y[:train_split]
x_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=x_train, train_labels=y_train, test_data=x_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size":14})
    plt.show()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        return self.weights * x + self.bias

torch.manual_seed(42)
model_0 = LinearRegressionModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

def train_test_loop(epochs):
   
    train_loss_history = []
    test_loss_history = []
    epoch_count = []

    for epoch in range(epochs):
        
        # Training
        model_0.train()
        y_pred = model_0(x_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Testing
        model_0.eval()
        with torch.inference_mode():
            test_pred = model_0(x_test)
            test_loss = loss_fn(test_pred, y_test)
        
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_history.append(loss.detach().numpy())
            test_loss_history.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
        
    return train_loss_history, test_loss_history, epoch_count

train_loss_history, test_loss_history, epoch_count = train_test_loop(100)
plt.plot(epoch_count, train_loss_history, label="Train loss")
plt.plot(epoch_count, test_loss_history, label="Test loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

def inference(model_0, x_test):
    model_0.eval()

    with torch.inference_mode():
        y_pred = model_0(x_test)

    plot_predictions(predictions=y_pred)

def export_model():
    from pathlib import Path
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "linear_regression.pth"
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
    print(f"Saved model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

def load_model(MODEL_SAVE_PATH):
    loaded_model_0 = LinearRegressionModel()
    loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    return loaded_model_0

inference(model_0, x_test)
export_model()

