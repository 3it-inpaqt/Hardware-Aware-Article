import os
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from dataset.moon_dataset import generate_moon
from networks.feed_forward import FeedForward
from networks.hardaware_feed_forward import Hardaware_FeedForward
from plots.misc import plot_fn
from test_standard import test_standard
from train_standard import train_standard
from utils.logger import logger
from utils.settings import settings

network_dict = {1: Hardaware_FeedForward, 2: FeedForward}
name_network_dict = {1: "Hardaware", 2: "FeedForward"}
network_type_dict = {"HANN": 1, "Modified HANN": 2}  # New dictionary to map network names to types
torch_device = torch.device("cpu")

# Number of simulations
n_simulations = 10

# Placeholder for accuracies and min_accuracies
accuracies = {"HANN": [], "Modified HANN": []}
min_accuracies = {"HANN": [], "Modified HANN": []}  # Store all minimum accuracies for each simulation

def main():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    if settings.generate_new_moon:  # If we want to generate a new moon
        trainset, testset, validationset = generate_moon()
        torch.save(trainset, settings.train_moon_dataset_location)
        torch.save(testset, settings.test_moon_dataset_location)
        torch.save(validationset, settings.validation_moon_dataset_location)
    else:  # Loading the dataset
        trainset = torch.load(settings.train_moon_dataset_location)
        testset = torch.load(settings.test_moon_dataset_location)
        validationset = torch.load(settings.validation_moon_dataset_location)

    settings.bayesian_complexity_cost_weight = 1 / (trainset.__len__())

    for network_type in [1, 2]:  # for both HANN and NN
        logger.info("Selected network: " + name_network_dict[network_type])
        criterion = torch.nn.BCEWithLogitsLoss()  # Define criterion outside of the if-statement
        if network_type == 1:
            network = network_dict[network_type](2, 1, elbo=settings.elbo)
        else:
            network = network_dict[network_type](2, 1)
        network.to(torch_device)

        # Pass network type to train_standard and test_standard
        # train_standard(network, trainset, testset, validationset, torch_device, network_type)

    compare_networks(criterion, testset)

    plot_cdf()

def compare_networks(criterion, testset):
    # Load the paths for each of the trained networks
    nn_path = settings.pretrained_address_dict[2]
    hann_path = settings.pretrained_address_dict[1]

    # Load the NN and HANN
    nn = torch.load(str(nn_path))
    hann = torch.load(str(hann_path))
    
    # Run the simulations
    for _ in range(n_simulations):
        # Create a copy of HANN
        hann_cpy = torch.load(str(hann_path))

        with torch.no_grad():
            # Copy the weights and bias of the first fully-connected layer from NN to HANN
            hann.fc1.weight = nn.fc1.weight
            hann.fc1.bias = nn.fc1.bias
            # Copy the weights and bias of the second fully-connected layer from NN to HANN
            hann.fc2.weight = nn.fc2.weight
            hann.fc2.bias = nn.fc2.bias

        # Test the original HANN
        network = hann_cpy.to(torch_device)
        acc, _ = test_standard(network, testset, torch_device, criterion, test_name='', network_type=1)
        accuracies["HANN"].append(acc)
        min_accuracies["HANN"].append(min(accuracies["HANN"]))

        # Test the modified HANN
        network = hann.to(torch_device)
        acc, _ = test_standard(network, testset, torch_device, criterion, test_name='', network_type=1)
        accuracies["Modified HANN"].append(acc)
        min_accuracies["Modified HANN"].append(min(accuracies["Modified HANN"]))

    return accuracies

def plot_cdf():
    bins = sorted([100, 95, 90, 80, 70, 60, 50, 0])  # the boundaries for the accuracy ranges
    bin_labels = ['100', '95 ≤ x < 100', '90 ≤ x < 95', '80 ≤ x < 90', '70 ≤ x < 80', '60 ≤ x < 70', '50 ≤ x < 60', 'x < 50']
    table = {label: {name: 0 for name in name_network_dict.values()} for label in bin_labels}  # initialize table

    for network_name, min_acc_list in min_accuracies.items():
        print(f"Min accuracies for {network_name}: {min_acc_list}")
        values, base = np.histogram(min_acc_list, bins=10)
        cumulative = np.cumsum(values)
        plt.plot(base[:-1], cumulative, label=name_network_dict[network_type_dict[network_name]])  # Use network_type_dict to map name to type

        # calculate the number of simulations falling within each accuracy range
        counts, _ = np.histogram(min_acc_list, bins=bins)
        cumulative = np.cumsum(counts[::-1])[::-1]  # Flip counts to get the right order for cumsum, then flip back
        cumulative = cumulative / cumulative[0] * 100  # Normalize to the total number of networks
        
        plt.plot(bins[:-1], cumulative, label=network_name)  # Removed the usage of network_type_dict

        for label, count in zip(bin_labels, counts):
            table[label][name_network_dict[network_type_dict[network_name]]] = count  # Use network_type_dict to map name to type

    plt.legend()
    plt.show()

    # print table
    print("Relative accuracies of every data point in the test set over {} simulated transfers:".format(n_simulations))
    print("Percentage of simulated networks that correctly classify the datapoints(%):")
    print("\t\t\t", "\t".join(name_network_dict.values()))
    for label, counts in table.items():
        print(label, "\t", "\t".join(str(count) + " (" + str(round(count / n_simulations * 100, 1)) + "%)" for count in counts.values()))

if __name__ == '__main__':
    main()