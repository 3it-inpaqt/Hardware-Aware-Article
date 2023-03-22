import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def generate_moon():
    X_old, Y_old = make_moons(noise=0.05, random_state=42, n_samples=1000)
    X= MinMaxScaler().fit_transform(X_old)
    X = torch.Tensor(X).float()
    Y = torch.Tensor(Y_old).float()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
    trainset = torch.utils.data.TensorDataset(X_train, Y_train)
    testset = torch.utils.data.TensorDataset(X_test, Y_test)
    fig, ax = plt.subplots()
    ax.scatter(X[Y_old == 0, 0], X[Y_old == 0, 1], label="Class 0")
    ax.scatter(X[Y_old == 1, 0], X[Y_old == 1, 1], color="r", label="Class 1")
    ax.legend()
    ax.set(xlabel="X", ylabel="Y", title="Toy binary classification data set");
    fig.show()
    return trainset, testset
