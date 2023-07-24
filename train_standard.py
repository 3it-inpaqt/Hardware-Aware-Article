from typing import List

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

from plots.misc import plot_train_progress, plot_weight_distribution, plot_confusion_matrix
from test_standard import test_standard
from utils.logger import logger
from utils.settings import settings
from utils.timer import SectionTimer
from tqdm import tqdm

def train_standard(network: Module, train_dataset: Dataset, test_dataset: Dataset, validation_dataset: Dataset, device: torch.device, network_type: int) -> None:
    network.train()

    # Define loaders
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=0)
    nb_batch = len(train_loader)

    # Other definitions
    checkpoints_i = [int(i / settings.checkpoints_per_epoch * nb_batch) for i in range(settings.checkpoints_per_epoch)]
    loss_evolution = []
    validation_loss_evolution = []
    accuracy_evolution = []
    epochs_stats = []
    epoch_accuracies = []
    epoch_accuracy_evolution = []

    total_iterations = settings.nb_epoch * nb_batch
    global_progress_bar = tqdm(total=total_iterations, desc='Training Progress', dynamic_ncols=True)

    with SectionTimer('network training') as timer:
        for epoch in range(settings.nb_epoch):
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                loss = network.training_step(inputs, labels)
                running_loss += loss.item()
                loss_evolution.append(float(loss))

                # Calculate accuracy for the current batch and append it to epoch_accuracies
                if network_type == 1:
                    outputs = network(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                elif network_type == 2:
                    predicted = network.infer(inputs).squeeze()

                correct_predictions = (predicted == labels).sum().item()
                accuracy = correct_predictions / labels.size(0)
                epoch_accuracies.append(accuracy)

                # Add batch_num, train_accuracy, and test_accuracy to accuracy_evolution for each batch
                if i in checkpoints_i:
                    network.eval()
                    all_labels, all_predictions = [], []
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            outputs = network(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            all_labels.extend(labels.numpy())
                            all_predictions.extend(predicted.numpy())

                    test_correct_predictions = (np.array(all_labels) == np.array(all_predictions)).sum()
                    test_accuracy = test_correct_predictions / len(all_labels)
                    accuracy_evolution.append({
                        'batch_num': i + epoch * nb_batch,
                        'test_accuracy': test_accuracy,
                        'train_accuracy': accuracy,
                    })

                global_progress_bar.set_postfix({'Loss': loss.item()})
                global_progress_bar.update()

            average_epoch_accuracy = np.mean(epoch_accuracies)  # Calculate the average accuracy of the epoch
            epoch_accuracy_evolution.append(average_epoch_accuracy)  # Append the value to the evolution list
            epoch_accuracies = []  # Reset the epoch accuracies for the next epoch

            _record_epoch_stats(epochs_stats, loss_evolution[-len(train_loader):], [average_epoch_accuracy])  # Pass the average accuracy of the epoch

            average_loss = running_loss / len(train_loader)
            total_validation_loss = 0

            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                validation_loss = network.validation_step(inputs, labels)
                total_validation_loss += validation_loss.item()

            average_validation_loss = total_validation_loss / len(validation_loader)
            network.train()
            validation_loss_evolution.append(average_validation_loss)

            if epoch in checkpoints_i:
                timer.pause()
                all_labels, all_predictions = [], []
                network.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = network(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        all_labels.extend(labels.numpy())
                        all_predictions.extend(predicted.numpy())

                # Generate confusion matrix and normalize it
                cm = confusion_matrix(all_labels, all_predictions)
                cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(float).eps)

                plot_confusion_matrix(cm_normalized)

                accuracy_evolution.append(_checkpoint(network, epoch, train_dataset, test_dataset, device))
                timer.resume()

    global_progress_bar.close()
    network.train()
    # print(accuracy_evolution)
    # plot_weight_distribution(network)
    # plot_train_progress(loss_evolution, validation_loss_evolution, accuracy_evolution, epoch_accuracy_evolution)


def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def _checkpoint(network: Module, batch_num: int, train_dataset: Dataset, test_dataset: Dataset, device, loss_fn) -> dict:
    # Start tests
    test_accuracy, test_loss = test_standard(network, test_dataset, loss_fn, test_name='checkpoint test',
                                  limit=settings.checkpoint_test_size,
                                  device=device)
    train_accuracy, train_loss = test_standard(network, train_dataset, loss_fn, test_name='checkpoint train',
                                   limit=settings.checkpoint_train_size,
                                   device=device)
    # Set it back to train because it was switched during tests
    network.train()

    # logger.info(f'Checkpoint {batch_num:<6n} '
    #             f'| test accuracy: {test_accuracy:5.2%} '
    #             f'| train accuracy: {train_accuracy:5.2%}')

    return {
        'batch_num': batch_num,
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'train_loss': train_loss,
    }


def _record_epoch_stats(epochs_stats: List[dict], epoch_losses: List[float], epoch_accuracies: List[float]) -> None:
    stats = {
        'losses_mean': float(np.mean(epoch_losses)),
        'losses_std': float(np.std(epoch_losses)),
        'accuracies_mean': float(np.mean(epoch_accuracies)),  
        'accuracies_std': float(np.std(epoch_accuracies)),  
    }

    # Compute the loss and accuracy difference with the previous epoch
    stats['losses_mean_diff'] = 0 if len(epochs_stats) == 0 else stats['losses_mean'] - epochs_stats[-1]['losses_mean']
    stats['accuracies_mean_diff'] = 0 if len(epochs_stats) == 0 else stats['accuracies_mean'] - epochs_stats[-1]['accuracies_mean']  
    epochs_stats.append(stats)

    # Log stats
    epoch_num = len(epochs_stats)
    # logger.info(f"Epoch {epoch_num:3}/{settings.nb_epoch} ({epoch_num / settings.nb_epoch:7.2%}) "
    #             f"| loss: {stats['losses_mean']:.5f} "
    #             f"| diff: {stats['losses_mean_diff']:+.5f} "
    #             f"| std: {stats['losses_std']:.5f}"
    #             f"| accuracy: {stats['accuracies_mean']:.5f} " 
    #             f"| diff: {stats['accuracies_mean_diff']:+.5f} "  
    #             f"| std: {stats['accuracies_std']:.5f}")  