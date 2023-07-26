import os
import torch
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataset.moon_dataset import generate_moon
from networks.feed_forward import FeedForward
from networks.hardaware_feed_forward import Hardaware_FeedForward
from test_standard import test_standard
from test_standard import test_individual
from train_standard import train_standard
from utils.logger import logger
from utils.settings import settings

network_dict = {1: Hardaware_FeedForward, 2: FeedForward}
name_network_dict = {1: "Hardaware", 2: "FeedForward"}
network_type_dict = {"HANN": 1, "Modified HANN": 2}  # New dictionary to map network names to types
torch_device = torch.device("cpu")

def count_accuracies(acc_dict):
    # Define accuracy ranges
    ranges = [(0.95, 1.01), (0.9, 0.95), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0, 0.5)]
    
    # Initialize counts dictionary
    counts_dict = {r: 0 for r in ranges}
    
    # Count accuracies in each range
    for acc in acc_dict.values():
        for r in ranges:
            if r[0] <= acc < r[1]:  # No need to multiply by 100 since the accuracy is already in decimal form
                counts_dict[r] += 1
                break  # Break the loop once the correct range is found
                
    return counts_dict

def create_table(counts_dict, total):
    # Create DataFrame
    df = pd.DataFrame(columns=['Percentage of simulated networks that correctly classify the datapoints(%)', '#data points'])
    
    # Fill DataFrame
    for r in sorted(counts_dict.keys(), reverse=True):
        row_name = f"{int(r[0]*100)} ≤ x < {int(r[1]*100)}"
        df.loc[row_name] = [counts_dict[r], f"{counts_dict[r]} ({counts_dict[r]/total*100:.1f}%)"]
        
    # Add total row
    df.loc['Total'] = [total, f"{total} (100%)"]
    
    return df

def main():
    
    trainset = torch.load(settings.train_moon_dataset_location)
    testset = torch.load(settings.test_moon_dataset_location)
    settings.bayesian_complexity_cost_weight = 1 / (trainset.__len__())
    criterion = torch.nn.BCEWithLogitsLoss()
    compare_networks(criterion, testset)

def generate_cdf(counts_dict):
    # Initialize dictionary for CDF
    cdf_dict = {}
    
    # Generate CDF
    total = sum(counts_dict.values())
    cumulative_count = 0
    for r in sorted(counts_dict.keys(), reverse=True):
        midpoint = (r[0] + r[1]) / 2  # Use midpoint of range as accuracy
        cumulative_count += counts_dict[r]
        cdf_dict[midpoint] = cumulative_count / total * 100  # Multiply by 100 to get percentage
        
    return cdf_dict

def plot_cdf(cdf_dict):
    plt.figure(figsize=(10, 7))
    for network_type, cdf in cdf_dict.items():
        # Add (100%, 0%) to the cdf
        cdf = {1.0: 0, **cdf}
        x = [r * 100 for r in cdf.keys()] 
        y = list(cdf.values())
        plt.plot(x, y, label=network_type, linewidth=2, alpha=0.7)

    plt.title('Comparaison of the minimal accuracy over n simulated transfers achievable for the test set', fontsize=15)
    plt.ylabel('Teset Properly classified (%)', fontsize=12)
    plt.xlabel('Transfered networks (%)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim([0, 100])  # Adjust y-axis limits
    plt.xlim([60, 105])  # Adjust x-axis limits
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter()) 
    plt.show()

def compare_networks(criterion, testset):
    testset = torch.load(settings.test_moon_dataset_location)
    # Load the paths for each of the trained networks
    hann_path = settings.pretrained_address_dict[1]
    nn_path = settings.pretrained_address_dict[2]

    # Load the HANN and NN
    hann = torch.load(str(hann_path))
    nn = torch.load(str(nn_path))
    # Create a copy of HANN
    hann_cpy = copy.deepcopy(hann)
    testset = torch.load(settings.test_moon_dataset_location)
    with torch.no_grad():
        # Copy the weights and bias of the first fully-connected layer from NN to HANN
        hann_cpy.fc1.weight = nn.fc1.weight
        hann_cpy.fc1.bias = nn.fc1.bias
        # Copy the weights and bias of the second fully-connected layer from NN to HANN
        hann_cpy.fc2.weight = nn.fc2.weight
        hann_cpy.fc2.bias = nn.fc2.bias

    # Test the original HANN
    network = hann.to(torch_device)
    hann_acc = test_individual(network, testset, torch_device, network_type=1)  
    # Count HANN accuracies
    hann_counts = count_accuracies(hann_acc)
    
    # Test the modified HANN (which is now a NN)
    network = hann_cpy.to(torch_device)
    hann_mod_acc = test_individual(network, testset, torch_device, network_type=1)   
    # Count modified HANN accuracies
    hann_mod_counts = count_accuracies(hann_mod_acc)
    
    # Create tables
    hann_table = create_table(hann_counts, len(testset))
    hann_mod_table = create_table(hann_mod_counts, len(testset))
    
    # Print tables
    print("HANN table:\n", hann_table)
    print("\nModified HANN table:\n", hann_mod_table)
    # Generate CDFs
    hann_cdf = generate_cdf(hann_counts)
    hann_mod_cdf = generate_cdf(hann_mod_counts)
    
    # Plot CDFs
    plot_cdf({"HANN": hann_cdf, "Modified HANN": hann_mod_cdf})

if __name__ == '__main__':
    main()