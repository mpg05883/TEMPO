import sys

import numpy as np
import torch
from numpy.random import choice
from torch.utils.data import Subset

from tempo.data_provider.data_factory import data_provider

SEED = 2021
np.random.seed(SEED)


def _update_args_from_config(args, config, dataset_name):
    """
    Updates the command line arguments with dataset-specific configurations
    from config.
    """

    # Get the configuration for the specified dataset name
    dataset_config = config["datasets"][dataset_name]
    
    keys = [
        "data",
        "root_path",
        "data_path",
        "data_name",
        "features",
        "freq",
        "target",
        "embed",
        "percent",
        "lradj",
    ]

    # Update args with the corresponding values from dataset_config
    for key in keys:
        setattr(args, key, getattr(dataset_config, key))

    # If frequency is set to 0, then set it to h (hourly)
    if args.freq == 0:
        args.freq = "h"

    # ? does this operate on args in place? or should args be returned?
    return args


def _combine_datasets(datasets):
    """
    Combine multiple datasets into one
    """
    combined = datasets[0]
    for dataset in datasets[1:]:
        combined = torch.utils.data.ConcatDataset([combined, dataset])
    return combined


def get_min_num_samples(args, config, train_dataset_names, excluded_datasets):
    """
    Iterates through all of the training datasets and determines which one has
    the smallest number of samples

    Args:
        train_dataset_names: A list of all training dataset names
        excluded_datasets: A set of dataset names to exclude from computing
                           min_num_samples

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Initialize to a very high value
    min_num_samples = sys.maxsize

    updated_args = args

    for dataset_name in train_dataset_names:
        # Update command line arguments using dataset-specific configurations
        # ? Does this return the updated args?
        updated_args = _update_args_from_config(args, config, dataset_name)

        # Load training set
        # TODO: read the code in data_provider
        train_data, _ = data_provider(args, "train")

        # If the current dataset should be excluded from equal sampling, then
        # jump to the next iteration
        if dataset_name in excluded_datasets:
            continue

        num_train_samples = len(train_data)

        # Update the minimum sample number
        min_sample_num = min(min_sample_num, num_train_samples)

    return min_num_samples, updated_args


def perform_equal_sampling(dataset_name, train_data, min_sample_num, args):
    # Number of samples in the training set
    num_samples = len(train_data)

    # Randomly select min_num_sample indices from num_samples
    selected_indices = choice(num_samples, min_sample_num)

    if dataset_name == "electricity" and args.electri_multiplier > 1:
        # Scale the minimum number of samples to use based on the multiplier
        scaled_min_sample_num = int(min_sample_num * args.electri_multiplier)

        # Randomly select scaled_min_sample_num indices from num_samples
        selected_indices = choice(num_samples, scaled_min_sample_num)

    elif dataset_name == "traffic" and args.traffic_multiplier > 1:
        # Scale the minimum number of samples to use based on the multiplier
        scaled_min_sample_num = int(min_sample_num * args.traffic_multiplier)

        # Randomly select scaled_min_sample_num indices from num_samples
        selected_indices = choice(num_samples, scaled_min_sample_num)

    # Get a subset of randomly selected elements from train_data
    return Subset(train_data, selected_indices)


def get_GluonTS_datasets(args, config):
    """
    Prepares and returns GluonTS datasets for training validation and testing.

    If multiple datasets are listed in args, then they're combined into one
    aggregate dataset.

    If args.equal_sampling is set to 1, then equal sampling is applied to each
    training dataset.

    Args:
        args: Arguments containing dataset configurations
        config: Configuration dictionary

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Initialize a list where each element is a different training dataset
    train_datasets = []

    # Initialize a list where each element is a different validation dataset
    val_datasets = []

    # Datasets to exclude from equal sampling
    excluded_datasets = {"ETTh1", "ETTh2", "ILI", "exchange", "monash"}

    # Names of the training datasets
    train_dataset_names = args.datasets.split(",")

    # Names of the validation datasets
    eval_dataset_names = args.eval_data.split(",")

    # Get the minimum number of samples to use from each dataset
    min_num_samples, args = get_min_num_samples(
        args,
        config,
        train_dataset_names,
        excluded_datasets,
    )

    # Get all training datasets
    for dataset_name in train_dataset_names:
        # Get current training set
        train_dataset, _ = data_provider(args, "train")

        # True if args.equal is set to 1 (i.e. we want equal sampling)
        equal_sampling = args.equal == 1

        # True if dataset_name is not in excluded_datasets
        not_excluded_dataset = dataset_name not in excluded_datasets

        if equal_sampling and not_excluded_dataset:
            train_dataset = perform_equal_sampling(
                dataset_name,
                train_dataset,
                min_num_samples,
                args,
            )

        # Add current training set to list of training datasets
        train_datasets.append(train_dataset)

    # Get all validation datasets
    for dataset_name in eval_dataset_names:
        # Get current validation set
        val_dataset, _ = data_provider(args, "val")

        # Add current validation set to list of validation datasets
        val_datasets.append(val_dataset)

    # True if there's more than one training set
    multiple_datasets = len(train_datasets) > 1

    # If there are multiple datasets, combine them into one aggregated dataset
    if multiple_datasets:
        train_data = _combine_datasets(train_datasets)
        val_data = _combine_datasets(val_datasets)
    else:
        train_data = train_datasets[0]
        val_data = val_datasets[0]

    # Get test set
    test_data, _ = data_provider(args, "test")

    # TODO: use train_data, val_data, and test_data to create GluonTS datasets

    # TODO: return GluonTS datasets
    return train_data, val_data, test_data
