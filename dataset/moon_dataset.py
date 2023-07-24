import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def generate_moon():
    # Generate data
    x_old, y_old = make_moons(noise=0.05, random_state=42, n_samples=1000)
    x = MinMaxScaler().fit_transform(x_old)
    x = torch.Tensor(x).float()
    y = torch.Tensor(y_old).float()

    # Split data into training, validation, and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    # Create datasets
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    validationset = torch.utils.data.TensorDataset(x_val, y_val)

    # Visualize datasets
    datasets = [(x_train, y_train, 'Training'), (x_test, y_test, 'Testing'), (x_val, y_val, 'Validation')]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, data in zip(axes, datasets):
        x, y, title = data
        ax.scatter(x[y == 0, 0], x[y == 0, 1], label="Class 0")
        ax.scatter(x[y == 1, 0], x[y == 1, 1], color="r", label="Class 1")
        ax.legend()
        ax.set(xlabel="X", ylabel="Y", title=f"{title} set")

    plt.tight_layout()
    plt.show()

    return trainset, testset, validationset
