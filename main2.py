import os
import torch
from pathlib import Path
from dataset.moon_dataset import generate_moon
from networks.feed_forward import FeedForward
from networks.hardaware_feed_forward import Hardaware_FeedForward
from plots.misc import plot_fn
from test_standard import test_standard
from train_standard import train_standard
from utils.logger import logger
from utils.settings import settings
from datetime import datetime

network_dict = {1: Hardaware_FeedForward, 2: FeedForward}
name_network_dict = {1: "Hardaware", 2: "FeedForward"}
torch_device = torch.device("cpu")

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
    logger.info("Selected network: " + name_network_dict[settings.choice])
    criterion = torch.nn.BCEWithLogitsLoss()  # Define criterion outside of the if-statement

    network_type = settings.choice

    if settings.choice == 1:
        network = network_dict[settings.choice](2, 1, elbo=settings.elbo)
    else:
        network = network_dict[settings.choice](2, 1)
    network.to(torch_device)

    # Pass network type to train_standard and test_standard
    train_standard(network, trainset, testset, validationset, torch_device, network_type)
    acc, _ = test_standard(network, testset, torch_device, criterion, test_name='', network_type=network_type)

    if settings.save_network:
        # Save the network to the appropriate location based on the network type
        network_dir = Path(os.getcwd(), "trained_networks")
        network_dir.mkdir(parents=True, exist_ok=True)
        network_filename = f"{name_network_dict[network_type]}_{str(timestamp).replace('.', '')}.pt"
        torch.save(network, network_dir / network_filename)

    # Call compare_networks() after training and testing
    criterion = torch.nn.BCEWithLogitsLoss()  # Define criterion outside of the if-statement
    network_type = settings.choice
    # compare_networks(criterion, network_type)

def compare_networks(criterion, network_type: int):
    # Load the paths for each of the trained networks
    nn_path = settings.pretrained_address_dict[2]
    hann_path = settings.pretrained_address_dict[1]
    hann_cpy_path = settings.pretrained_address_dict[1]
    nn = torch.load(str(nn_path))
    hann = torch.load(str(hann_path))
    hann_cpy = torch.load(str(hann_cpy_path))

    # Load the test dataset
    testset = torch.load(settings.test_moon_dataset_location)
    with torch.no_grad():
        # Copy the weights and bias of the first fully-connected layer
        hann.fc1.weight = nn.fc1.weight
        hann.fc1.bias = nn.fc1.bias
        # Copy the weights and bias of the second fully-connected layer
        hann.fc2.weight = nn.fc2.weight
        hann.fc2.bias = nn.fc2.bias
    # Test the copied hardware-aware neural network
    if settings.choice == 1:
        network = network_dict[settings.choice](2, 1, elbo=settings.elbo)
    else:
        network = network_dict[settings.choice](2, 1)
    network.to(torch_device)
    acc, _ = test_standard(network, testset, torch_device, criterion, test_name='', network_type=network_type)

    print(f"Accuracy of the modified hardware-aware network: {acc}")

if __name__ == '__main__':
    main()
