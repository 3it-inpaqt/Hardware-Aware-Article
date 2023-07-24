import functools
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LogisticRegression
from utils.settings import settings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 2

mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1.5

mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 1.5


def plot_train_progress(loss_evolution: List[float], validation_loss_evolution: List[float], accuracy_evolution: List[dict],
                        epoch_accuracy_evolution: List[float], batch_per_epoch: int = 0) -> None:
    """
    Plot the evolution of the loss, validation loss, and the accuracy during the training.
    
    :param loss_evolution: A list of loss for each batch.
    :param validation_loss_evolution: A list of validation loss for each epoch.
    :param accuracy_evolution: A list of dictionaries as {batch_num, test_accuracy, train_accuracy}.
    :param epoch_accuracy_evolution: A list of average accuracy for each epoch.
    :param batch_per_epoch: The number of batch per epoch to plot x ticks.
    """
    color_dict = {'train loss': 'tab:gray', 'validation loss': 'tab:blue', 'test accuracy': 'tab:green', 'train accuracy': 'tab:orange', 'epoch accuracy': 'tab:red'}
    with sns.axes_style("ticks"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Vertical lines for each batch
        if batch_per_epoch:
            for epoch in range(0, len(loss_evolution) + 1, batch_per_epoch):
                ax1.axvline(x=epoch, color='black', linestyle=':', alpha=0.2, label='epoch' if epoch == 0 else '')

        # Plot loss
        ax1.plot(loss_evolution, label='train loss', color=color_dict['train loss'])
        ax1.set_ylabel('Loss')
        ax1.set_ylim(bottom=0)

        # Plot validation loss
        ax2.plot(validation_loss_evolution, label='validation loss', color=color_dict['validation loss'])
        ax2.set_ylabel('Validation Loss')
        ax2.set_xlabel('Epoch')
        if accuracy_evolution:
            # Check if the list is not empty and the necessary keys exist
            if accuracy_evolution and all(key in accuracy_evolution[0] for key in ['batch_num', 'test_accuracy', 'train_accuracy']):
                checkpoint_batches = [a['batch_num'] for a in accuracy_evolution]
                test_accuracies = [a['test_accuracy'] for a in accuracy_evolution]
                train_accuracies = [a['train_accuracy'] for a in accuracy_evolution]

                # Check if the data can be converted to float
                try:
                    test_accuracies = [float(a) for a in test_accuracies]
                    train_accuracies = [float(a) for a in train_accuracies]
                except ValueError:
                    print("Error: Accuracy data could not be converted to float.")
                    return

                # Check if the data is within the correct range
                # if not all(0 <= a <= 1 for a in test_accuracies + train_accuracies):
                #     print("Error: Accuracy data is not within the range 0-1.")
                #     return

                # If all checks pass, plot the accuracy evolution
                # ax3.plot(checkpoint_batches, test_accuracies,
                #         label='test accuracy', color=color_dict['test accuracy'])
                # ax3.plot(checkpoint_batches, train_accuracies,
                #         label='train accuracy', color=color_dict['train accuracy'])
                # ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
                # ax3.set_ylabel('Accuracy')
                # ax3.set_ylim(bottom=0, top=1)
        # else:
        #     print("No accuracy data available.")

        # Plot epoch accuracy
        # ax4.plot(epoch_accuracy_evolution, label='epoch accuracy', color=color_dict['epoch accuracy'])
        # ax4.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        # ax4.set_ylabel('Epoch Accuracy')
        # ax4.set_xlabel('Epoch')

        # Add titles and labels
        ax1.set_title('Training Loss')
        ax2.set_title('Validation Loss')
        # ax3.set_title('Accuracy Evolution')
        # ax4.set_title('Epoch Accuracy Evolution')
        # ax4.set_xlabel(f'Batch number (size: {settings.batch_size:n})')

        # Place legends
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper left")
        # ax3.legend(loc="upper left")
        # ax4.legend(loc="upper left")

        plt.tight_layout()  # adjust subplot parameters to give specified padding

        plt.show()


#        save_plot('train_progress')

def plot_confusion_matrix(nb_labels_predictions: np.ndarray, class_names: List[str] = None,
                          annotations: bool = True) -> None:
    """
    Plot the confusion matrix for a set a predictions.

    :param nb_labels_predictions: The count of prediction for each label.
    :param class_names: The list of readable classes names
    :param annotations: If true the accuracy will be written in every cell
    """

    overall_accuracy = nb_labels_predictions.trace() / nb_labels_predictions.sum()
    rate_labels_predictions = nb_labels_predictions / (nb_labels_predictions.sum(axis=1).reshape((-1, 1)) + np.finfo(float).eps)
    plt.figure()
    sns.heatmap(rate_labels_predictions,
                vmin=0,
                vmax=1,
                square=True,
                fmt='.1%',
                cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                annot=annotations,
                cbar=(not annotations))
    plt.title(f'Confusion matrix of {len(nb_labels_predictions)} classes '
              f'with {overall_accuracy * 100:.2f}% overall accuracy')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.show()
    
from sklearn.svm import SVC

def plot_fn(train, test, validation, point_size=4, title=None, xlabel=None, ylabel=None):
    """
    Function to plot training, test, and validation datasets.

    Parameters
    ----------
    train : tuple
        Training set (coordinates, labels).
    test : tuple
        Test set (coordinates, labels).
    validation : tuple
        Validation set (coordinates, labels).
    point_size : int, optional
        Size of the points in the scatter plot.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    None
    """

    if train is not None:
        train_coords, train_labels = train
        train_xs = train_coords[:, 0]
        train_ys = train_coords[:, 1]
        plt.scatter(train_xs, train_ys, s=point_size, label='train')

    if test is not None:
        test_coords, test_labels = test
        test_xs = test_coords[:, 0]
        test_ys = test_coords[:, 1]
        plt.scatter(test_xs, test_ys, s=point_size, label='test')

    if validation is not None:
        validation_coords, validation_labels = validation
        validation_xs = validation_coords[:, 0]
        validation_ys = validation_coords[:, 1]
        plt.scatter(validation_xs, validation_ys, s=point_size, label='validation')

    # Fit an SVM with an RBF kernel
    X = np.concatenate((train_coords, test_coords, validation_coords))
    y = np.concatenate((train_labels, test_labels, validation_labels))
    model = SVC(kernel='rbf')
    model.fit(X, y)

    # Generate grid points to evaluate the decision function
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict the class labels for the grid points
    decision_values = model.predict(grid_points)
    decision_values = decision_values.reshape(xx.shape)

    # Plot the decision boundary
    plt.contour(xx, yy, decision_values, levels=[0], colors='red', linestyles='dashed', label='decision boundary')

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.legend()
    plt.show()

    # Assuming `format_plot` is a function you have defined elsewhere in your code
    format_plot(xlabel, ylabel)


def format_plot(x=None, y=None):
    if x is not None:
        plt.xlabel(x, fontsize=20)
    if y is not None:
        plt.ylabel(y, fontsize=20)


def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
    plt.tight_layout()
    legend = functools.partial(plt.legend, fontsize=10)


def plot_weight_distribution(network):
    """
    Util function used to plot the distribution of the weights in the network
    Parameters
    ----------
    network: The Neural Network in question
    """
    all_weights = []
    for param in network.parameters():
        all_weights.extend(flatten_list(param.tolist()))
    all_weights = np.array(all_weights)
    mean = np.mean(all_weights)
    std = np.std(all_weights)
    fig = plt.figure()
    ax = fig.add_subplot()
    domain = np.linspace(mean - 3 * std, mean + 3 * std)
    ax.plot(domain, stats.norm.pdf(domain, mean, std), color="#1f77b4")
    ax.hist(all_weights, edgecolor='black', alpha=.5, bins=20, density=True, color="green")
    plt.grid()
    title = "Distribution of weights"
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    plt.show()
    return


def flatten_list(_2d_list):
    """Flattens a list"""
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def plot_output_and_training_data(train_dataset, network):
    n_grid = 200
    xy = torch.meshgrid(torch.linspace(0,1,n_grid), torch.linspace(0,1,n_grid), indexing='xy')
    xy = torch.stack(xy).reshape(2,-1).T
    out = torch.sigmoid(network(xy)).squeeze().reshape((n_grid,n_grid))

    train_points_0 = torch.stack([xy for xy, label in train_dataset if label==0]).detach().numpy()
    train_points_1 = torch.stack([xy for xy, label in train_dataset if label==1]).detach().numpy()

    plt.imshow(out.detach().numpy(), interpolation='nearest', aspect='auto',
               origin='lower', cmap='viridis_r', extent=(0,1,0,1))
    plt.colorbar()
    plt.scatter(train_points_0[:,0], train_points_0[:,1], color='g', edgecolors='k', s=20)
    plt.scatter(train_points_1[:,0], train_points_1[:,1], color='cornflowerblue', edgecolors='k', s=20)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show(block=True)


def plot_uncertainty_predicted_value(all_inputs, network):
    l = 1  # limit
    grid = np.mgrid[-l:l:200j, -l:l:200j]
    grid_2d = grid.reshape(2, -1).T
    outputs = network.infer(torch.tensor(grid_2d).float(), 1000)
    network.layers[0].no_variability = True
    network.layers[1].no_variability = True
    cmap = "viridis_r"  # sns.cubehelix_palette(light=1, as_cmap=True)
    testset = torch.load(settings.test_moon_dataset_location)

    # Uncertainty values
    uncertainty_values = outputs[1][1].detach().numpy().reshape(200, 200)
    vmin_uncertainty, vmax_uncertainty = np.min(uncertainty_values), np.max(uncertainty_values)

    # Mean values
    mean_values = outputs[1][0].detach().numpy().reshape(200, 200)
    vmin_mean, vmax_mean = np.min(mean_values), np.max(mean_values)

    # Uncertainty plot
    fig, ax = plt.subplots(figsize=(16, 9))
    contour = ax.contourf(grid[0], grid[1], uncertainty_values, cmap=cmap, vmin=vmin_uncertainty, vmax=vmax_uncertainty)
    ax.scatter(testset.tensors[0][testset.tensors[1] == 1][:, 0], testset.tensors[0][testset.tensors[1] == 1][:, 1],
               color="g", edgecolors="k")
    ax.scatter(testset.tensors[0][testset.tensors[1] == 0][:, 0], testset.tensors[0][testset.tensors[1] == 0][:, 1],
               color="cornflowerblue", edgecolors="k")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set(xlim=(0, l), ylim=(0, l), xlabel="X", ylabel="Y")
    cbar = plt.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel("Standard deviation on simulated transfers")
    plt.title("Uncertainty in Predicted Values")

    out_dir = Path('out/saved_plots')
    out_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(Path(out_dir, 'uncertainty_solution.png'), format='png', dpi=400, bbox_inches='tight')

    # Mean values plot
    fig, ax = plt.subplots(figsize=(16, 9))
    contour = ax.contourf(grid[0], grid[1], mean_values, cmap=cmap, vmin=vmin_mean, vmax=vmax_mean)
    ax.scatter(testset.tensors[0][testset.tensors[1] == 1][:, 0], testset.tensors[0][testset.tensors[1] == 1][:, 1],
               color="g", edgecolors="k")
    ax.scatter(testset.tensors[0][testset.tensors[1] == 0][:, 0], testset.tensors[0][testset.tensors[1] == 0][:, 1],
               color="cornflowerblue", edgecolors="k")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set(xlim=(0, l), ylim=(0, l), xlabel="X", ylabel="Y")
    cbar = plt.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel("Mean value on simulated transfers")
    plt.title("Mean values in Predicted Values")

    plt.savefig(Path(out_dir, 'mean_solution.png'), format='png', dpi=400, bbox_inches='tight')

    # # Compute average uncertainty
    # avg_uncertainty = uncertainty_values.mean()
    # # Compute standard deviation of uncertainty
    # std_uncertainty = uncertainty_values.std()

    # # Now let's plot histogram of uncertainty
    # fig, ax = plt.subplots(figsize=(10, 6))
    # counts, bins, patches = ax.hist(uncertainty_values.flatten(), bins=50, color='blue', alpha=0.7)
    # ax.set(xlabel='Uncertainty', ylabel='Frequency')
    # plt.title("Histogram of Uncertainty in Predicted Values")  # Adding a title
    # plt.grid(True, linestyle='--', alpha=0.6)  # Adding gridlines

    # # Add the metrics to the histogram as well
    # bin_centers = 0.5 * (bins[:-1] + bins[1:])  # center of each bin
    # metrics_y_position = bin_centers[-1]

    # ax.text(0.95, 0.95, f'Average uncertainty: {avg_uncertainty:.2f}',
    #         horizontalalignment='right',
    #         verticalalignment='top',
    #         transform=ax.transAxes,
    #         color='black', fontsize=10)

    # ax.text(0.95, 0.90, f'Standard deviation of uncertainty: {std_uncertainty:.2f}',
    #         horizontalalignment='right',
    #         verticalalignment='top',
    #         transform=ax.transAxes,
    #         color='black', fontsize=10)

    # plt.savefig(Path(out_dir, 'uncertainty_histogram.png'), format='png', dpi=400, bbox_inches='tight')

    # # Histogram of mean values
    # fig, ax = plt.subplots(figsize=(10, 6))
    # counts, bins, patches = ax.hist(mean_values.flatten(), bins=50, color='blue', alpha=0.7)
    # ax.set(xlabel='Mean Value', ylabel='Frequency')
    # plt.title("Histogram of Mean Values in Predicted Values")  # Adding a title
    # plt.grid(True, linestyle='--', alpha=0.6)  # Adding gridlines

    # avg_mean = mean_values.mean()
    # std_mean = mean_values.std()

    # ax.text(0.95, 0.95, f'Average Mean: {avg_mean:.2f}',
    #         horizontalalignment='right',
    #         verticalalignment='top',
    #         transform=ax.transAxes,
    #         color='black', fontsize=10)

    # ax.text(0.95, 0.90, f'Standard Deviation of Mean: {std_mean:.2f}',
    #         horizontalalignment='right',
    #         verticalalignment='top',
    #         transform=ax.transAxes,
    #         color='black', fontsize=10)

    # plt.savefig(Path(out_dir, 'mean_histogram.png'), format='png', dpi=400, bbox_inches='tight')

    plt.show()

