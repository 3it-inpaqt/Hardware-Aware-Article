import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def generate_moon():
    x_old, y_old = make_moons(noise=0.05, random_state=42, n_samples=1000)
    x = MinMaxScaler().fit_transform(x_old)
    x = torch.Tensor(x).float()
    y = torch.Tensor(y_old).float()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    fig, ax = plt.subplots()
    ax.scatter(x[y_old == 0, 0], x[y_old == 0, 1], label="Class 0")
    ax.scatter(x[y_old == 1, 0], x[y_old == 1, 1], color="r", label="Class 1")
    ax.legend()
    ax.set(xlabel="X", ylabel="Y", title="Toy binary classification data set")
    fig.show()
    return trainset, testset
