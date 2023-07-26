import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from typing import Callable
from typing import Tuple
from utils.settings import settings
from plots.misc import plot_uncertainty_predicted_value
from plots.misc import plot_output_and_training_data
from utils.logger import logger
from utils.settings import settings
from utils.timer import SectionTimer
import numpy as np

def test_standard(network: Module, test_dataset: Dataset, device: torch.device, criterion, network_type: int, test_name: str = '', final: bool = False, limit: int = 0) -> float:
    """
    Start testing the network on a dataset.

    :param network: The network to use.
    :param test_dataset: The testing dataset.
    :param device: The device used to store the network and datasets (it can influence the behaviour of the testing)
    :param test_name: Name of this test for logging and timers.
    :param final: If true this is the final test, will show in log info and save results in file.
    :param limit: Limit of item from the dataset to evaluate during this testing (0 to run process the whole dataset).
    :return: The overall accuracy.
    """
    # Use the pyTorch data loader
    if test_name:
        test_name = ' ' + test_name

    nb_test_items = min(len(test_dataset), limit) if limit else len(test_dataset)
    logger.debug(f'Testing{test_name} on {nb_test_items:n} inputs')

    # Turn on the inference mode of the network
    network.eval()

    # Use the pyTorch data loader
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=0)
    nb_classes = 2  # len(test_dataset.classes) #TODO: CHECK HERE LEN TEST_DATASET
    nb_correct = 0  # counter of correct classifications
    nb_total = 0  # counter of total classifications
    # Create the tensors
    all_means = torch.Tensor()
    all_stds = torch.Tensor()
    all_inputs = torch.Tensor()
    all_outputs = torch.Tensor()
    # Disable gradient for performances
    running_loss = 0.0
    with torch.no_grad(), SectionTimer(f'network testing{test_name}', 'info' if final else 'debug'):
        for i, (inputs, labels) in enumerate(test_loader):
            # Stop testing after the limit
            if limit and i * settings.batch_size >= limit:
                break
            outputs = network.infer(inputs, settings.inference_number_contour)
            if settings.choice != 2:  # If Hardware-aware or Bayesian Network we need multiple inferences
                all_inputs = torch.cat((all_inputs, inputs))
                all_means = torch.cat((all_means, outputs[1][0]))
                all_stds = torch.cat((all_stds, outputs[1][1]))
                all_outputs = torch.cat((all_outputs, outputs[0]))
                nb_total += len(labels)
                nb_correct += int(torch.eq(outputs[0].flatten(), labels).sum())
            else:
                all_inputs = torch.cat((all_inputs, inputs))
                all_outputs = outputs[0].flatten()
                nb_total += len(labels)
                nb_correct += int(torch.eq(outputs.flatten(), labels).sum())
    # accuracy
    accuracy = float(nb_correct / nb_total)
    test_loss = running_loss / len(test_loader)
    testset = torch.load(settings.test_moon_dataset_location)
    actual_values = testset.tensors[1] 
    if network_type == 1:
        all_outputs = network.infer(all_inputs.float(), 1000) 
        uncertainty_values = all_outputs[1][1].detach().numpy()
        mean_values = all_outputs[1][0].detach().numpy()

        vmin_uncertainty = np.min(uncertainty_values)
        vmax_uncertainty = np.max(uncertainty_values)

        vmin_mean = np.min(mean_values)
        vmax_mean = np.max(mean_values)

        # plot_uncertainty_predicted_value(all_inputs, network)

    elif network_type == 2:
        all_outputs = torch.sigmoid(network(all_inputs)).detach().numpy() 
        vmin_value = np.min(all_outputs)
        vmax_value = np.max(all_outputs)

        plot_output_and_training_data(test_dataset, network)


    logger.info("mean accuracy: " + str(accuracy))
    return accuracy, test_loss

from tqdm import tqdm

def test_individual(network: Module, test_dataset: Dataset, device: torch.device, network_type: int, n_simulations: int = 10):
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    correct = {i: 0 for i in range(len(test_dataset))}  # A dictionary to store the number of correct predictions for each data point
    total = {i: 0 for i in range(len(test_dataset))}  # A dictionary to store the total number of predictions for each data point
    with torch.no_grad():
        for i in tqdm(range(n_simulations)):  # For each simulation
            for j, (inputs, labels) in enumerate(test_loader):  # For each data point in the batch
                outputs = network(inputs.to(device))
                if network_type == 1:  # If it's a Bayesian network
                    prediction = network.infer(inputs.to(device), settings.bayesian_nb_sample)[0]
                else:
                    prediction = torch.round(torch.sigmoid(outputs))
                total[j] += 1
                if prediction == labels.to(device):  # If the prediction is correct
                    correct[j] += 1
    # Calculate the relative accuracy for each data point
    relative_accuracies = {i: correct[i] / total[i] for i in range(len(test_dataset))}
    return relative_accuracies
