from dataset.moon_dataset import generate_moon
from networks.hardaware_feed_forward import Hardaware_FeedForward
from networks.feed_forward import FeedForward
from train_Standard import train_Standard
from test_Standard import test_Standard
from utils.logger import logger
from plots.misc import plot_fn
import torch
import os
from utils.settings import settings

network_dict = {1:Hardaware_FeedForward, 2: FeedForward}
name_network_dict = {1:"Hardaware",2:"FeedForward"}
def main():
    if settings.generate_new_moon: #If we want to generate a new moon
        trainset, testset, validationset = generate_moon()
        torch.save(trainset,settings.train_moon_dataset_location)
        torch.save(testset,settings.test_moon_dataset_location)
        torch.save(validationset,settings.validation_moon_dataset_location)
        #Plots the trainset
        plot_fn((trainset.tensors[0], trainset.tensors[1]), (testset.tensors[0], testset.tensors[1]))
    else: # Loading the dataset
        trainset = torch.load(settings.train_moon_dataset_location)
        testset = torch.load(settings.test_moon_dataset_location)
        validationset = torch.load(settings.validation_moon_dataset_location)
        #plot_fn((trainset.tensors[0],trainset.tensors[1]),(testset.tensors[0],testset.tensors[1]),(validationset.tensors[0],validationset.tensors[1]))
    settings.bayesian_complexity_cost_weight = 1/(trainset.__len__())
    logger.info("Selected network: " + name_network_dict[settings.choice])
    if settings.choice == 1:
        NN = network_dict[settings.choice](2, 1,elbo = settings.elbo)
    else:
        NN = network_dict[settings.choice](2, 1)
    train_Standard(NN, trainset, testset, "cpu")
    acc = test_Standard(NN, testset, "cpu")
    if settings.save_network:
        torch.save(NN, settings.pretrained_address)
    return

def Compare_networks():
    """
    Method used to insert the weights of the regular Neural Network into the Hardware-Aware framework.
    Returns
    -------
    """
    NN = torch.load(os.getcwd() + "\\good_trained_networks\\FF_complete_precision_new_dataset\\FF_166636344039526.pt")
    HANN = torch.load(os.getcwd() + "\\good_trained_networks\\HAFF-complete_precision-P464E-Subs_5LRS_2HRS_AE_FIXED\\HAFF_1667851816242543.pt")
    HANN_cpy = torch.load(os.getcwd() + "\\good_trained_networks\\HAFF-complete_precision-P464E-Subs_5LRS_2HRS_AE_FIXED\\HAFF_1667851816242543.pt")
    testset = torch.load(settings.test_moon_dataset_location)
    with torch.no_grad():
        HANN.fc1.weight = NN.fc1.weight
        HANN.fc1.bias = NN.fc1.bias
        HANN.fc2.weight = NN.fc2.weight
        HANN.fc2.bias = NN.fc2.bias
    acc = test_Standard(HANN, testset, "cpu")
    acc = test_Standard(HANN_cpy,testset,"cpu")
if __name__ == '__main__':
    #cross_validate()
    main()
