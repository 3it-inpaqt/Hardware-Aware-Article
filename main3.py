import os
import torch
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
torch_device = torch.device("cpu")


def main():
    if settings.generate_new_moon:  # If we want to generate a new moon
        trainset, testset, validationset = generate_moon()
        torch.save(trainset, settings.train_moon_dataset_location)
        torch.save(testset, settings.test_moon_dataset_location)
        torch.save(validationset, settings.validation_moon_dataset_location)
        # Plots the trainset
        plot_fn((trainset.tensors[0], trainset.tensors[1]), (testset.tensors[0], testset.tensors[1]))
    else:  # Loading the dataset
        trainset = torch.load(settings.train_moon_dataset_location)
        testset = torch.load(settings.test_moon_dataset_location)
        validationset = torch.load(settings.validation_moon_dataset_location)
        # plot_fn((trainset.tensors[0],trainset.tensors[1]),(testset.tensors[0],testset.tensors[1]),(validationset.tensors[0],validationset.tensors[1]))
    settings.bayesian_complexity_cost_weight = 1 / (trainset.__len__())
    logger.info("Selected network: " + name_network_dict[settings.choice])
    if settings.choice == 1:
        nn = network_dict[settings.choice](2, 1, elbo=settings.elbo)
    else:
        nn = network_dict[settings.choice](2, 1)
    # train_standard(nn, trainset, testset, torch_device)
    # acc = test_standard(nn, testset, torch_device)
    if settings.save_network:
        torch.save(nn, settings.pretrained_address)

    # Call compare_networks() function
    compare_networks()
    return


def compare_networks():
    """
    Method used to insert the weights of the regular Neural Network into the Hardware-Aware framework.
    Returns
    -------
    """
    # Load the paths for each of the trained networks
    nn_path = settings.pretrained_address_dict[2]
    hann_path = settings.pretrained_address_dict[1]

    # Load the NN and HANN
    nn = torch.load(str(nn_path))
    hann = torch.load(str(hann_path))
    hann_cpy = torch.load(str(hann_path))
    testset = torch.load(settings.test_moon_dataset_location)
    with torch.no_grad():
        hann.fc1.weight = nn.fc1.weight
        hann.fc1.bias = nn.fc1.bias
        hann.fc2.weight = nn.fc2.weight
        hann.fc2.bias = nn.fc2.bias
    acc = test_standard(hann, testset, torch_device)
    acc = test_standard(hann_cpy, testset, torch_device)
    print(acc)

if __name__ == '__main__':
    # cross_validate()
    main()
