import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from numpy.random import choice
from omegaconf import OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from smape import SMAPE
from tempo.data_provider.data_factory import data_provider
from tempo.models.DLinear import DLinear
from tempo.models.ETSformer import ETSformer
from tempo.models.GPT4TS import GPT4TS
from tempo.models.PatchTST import PatchTST
from tempo.models.T5 import T54TS
from tempo.models.TEMPO import TEMPO
from tempo.trainer.trainer import Trainer
from tempo.utils.tools import EarlyStopping, print_rank_0

FIX_SEED = 2021
random.seed(FIX_SEED)
torch.manual_seed(FIX_SEED)
np.random.seed(FIX_SEED)

SEASONALITY_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1,
}


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


def get_model(architecture: str = None):
    if architecture == "PatchTST":
        model = PatchTST(args)
    elif architecture == "DLinear":
        model = DLinear(args)
    elif architecture == "TEMPO":
        model = TEMPO(args)
    elif architecture == "T5":
        model = T54TS(args)
    elif "ETSformer" in architecture:
        model = ETSformer(args)
    else:
        model = GPT4TS(args)
    return model


def _combine_datasets(datasets):
    """Combine multiple datasets into one"""
    combined = datasets[0]
    for dataset in datasets[1:]:
        combined = torch.utils.data.ConcatDataset([combined, dataset])
    return combined


def _update_args_from_config(args, config, dataset_name):
    """Update args with dataset specific configurations"""
    dataset_config = config["datasets"][dataset_name]
    for key in [
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
    ]:
        setattr(args, key, getattr(dataset_config, key))

    if args.freq == 0:
        args.freq = "h"


def print_dataset_info(data, loader, name="Dataset"):
    print_rank_0(f"\n{name} Info:")
    print_rank_0(f"- Number of samples: {len(data):,}")
    print_rank_0(f"- Batch size: {loader.batch_size}")
    print_rank_0(f"- Number of batches: {len(loader)}")

    attributes = ["features", "targets", "shape"]
    for attr in attributes:
        if hasattr(data, attr):
            print_rank_0(f"- {attr}: {getattr(data, attr)}")


def prepare_data_loaders(args, config):
    """
    Prepares train, validation and test data loaders for use with one or more
    GPUs

    Args:
        args: Command line arguments
        config: Configuration object for model

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_datas = []
    val_datas = []
    min_sample_num = sys.maxsize

    # First pass to get validation data and minimum sample number
    for dataset_name in args.datasets.split(","):
        _update_args_from_config(args, config, dataset_name)

        train_data, train_loader = data_provider(args, "train")
        if dataset_name not in ["ETTh1", "ETTh2", "ILI", "exchange", "monash"]:
            min_sample_num = min(min_sample_num, len(train_data))

    for dataset_name in args.eval_data.split(","):
        _update_args_from_config(args, config, dataset_name)
        val_data, val_loader = data_provider(args, "val")
        val_datas.append(val_data)

    # Second pass to prepare training data with proper sampling
    for dataset_name in args.datasets.split(","):
        _update_args_from_config(args, config, dataset_name)

        train_data, _ = data_provider(args, "train")

        if (
            dataset_name not in ["ETTh1", "ETTh2", "ILI", "exchange", "monash"]
            and args.equal == 1
        ):
            train_data = Subset(train_data, choice(len(train_data), min_sample_num))

        if args.equal == 1:
            if dataset_name == "electricity" and args.electri_multiplier > 1:
                train_data = Subset(
                    train_data,
                    choice(
                        len(train_data), int(min_sample_num * args.electri_multiplier)
                    ),
                )
            elif dataset_name == "traffic" and args.traffic_multiplier > 1:
                train_data = Subset(
                    train_data,
                    choice(
                        len(train_data), int(min_sample_num * args.traffic_multiplier)
                    ),
                )

        train_datas.append(train_data)

    # Combine datasets if multiple exist
    if len(train_datas) > 1:
        train_data = _combine_datasets(train_datas)
        val_data = _combine_datasets(val_datas)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=DistributedSampler(train_data),
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=DistributedSampler(val_data),
        )

    # Prepare test data
    _update_args_from_config(args, config, args.target_data)
    test_data, test_loader = data_provider(args, "test")

    print_dataset_info(train_data, train_loader, "Training Dataset")
    print_dataset_info(val_data, val_loader, "Validation Dataset")
    print_dataset_info(test_data, test_loader, "Test Dataset")

    return train_loader, val_loader, test_loader


def get_checkpoint_path(loss_func: str):
    """
    Returns a file path to a trained model's checkpoint

    Args:
        loss_func: Loss function that model was trained to optimized.
                   If loss_func is "mse," then deterministic model's checkpoint
                   will be returned. Else, probabilistic model's checkpoint
                   will be returned

    Returns:
        checkpoint_path: File path to trained model's checkpoint
    """
    # Directory where checkpoints for all models are stored
    checkpoints_directory = "checkpoints"

    checkpoints_directory_path = os.path.join(checkpoints_directory, "Monash_1")

    is_prob = "_Prob_" if loss_func == "mse" else "_"

    # Directory where model's checkpoint is stored
    model_directory = f"Demo_Monash_TEMPO{is_prob}6_prompt_learn_336_96_100_sl336_ll0_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0"

    # Path to model_directory (./checkpoints/model_directory)
    model_directory_path = os.path.join(checkpoints_directory_path, model_directory)

    # Name of checkpoint file in model_directory
    checkpoint_file = "checkpoint.pth"

    # Path to checkpoint.pth (./checkpoints/model_directory/checkpoint_file)
    checkpoint_path = os.path.join(model_directory_path, checkpoint_file)

    return checkpoint_path


def studentT_nll(y_true, y_pred):
    """
    Loss function where the loss is calculated as the negative log-likelihood
    of a Student's t-distribution

    Args:
        y_true: List of true values
        y_pred: List of predicted values

    Returns:
        : Negative log-likelihood
    """
    y_true = y_true.squeeze()
    mu, sigma, nu = y_pred[0], y_pred[1], y_pred[2]

    # Create the Student's t-distribution
    nu = torch.abs(nu) + 1e-6
    sigma = torch.abs(sigma) + 1e-6
    mu = mu
    student_t = dist.StudentT(df=nu, loc=mu, scale=sigma)

    # Calculate the negative log-likelihood
    nll = -student_t.log_prob(y_true)
    return nll.mean()


def negative_binomial_nll(target, y_pred):
    """
    Loss function where the loss is calculated as the negative log-likelihood
    of a Negative Binomial distribution

    Args:
        target: List of true values
        y_pred: List of predicted values

    Returns:
        : Negative log-likelihood
    """
    # Compute negative log-likelihood of Negative Binomial distribution
    mu, alpha = y_pred[0], y_pred[1]
    if len(target.shape) != 3:
        target = target.unsqueeze(2)
    log_gamma_x_plus_n = torch.lgamma(target + 1.0 / alpha)
    log_gamma_x = torch.lgamma(target + 1)
    log_gamma_n = torch.lgamma(1.0 / alpha)

    log_prob = (
        log_gamma_x_plus_n
        - log_gamma_x
        - log_gamma_n
        - (target + 1.0 / alpha) * torch.log1p(alpha * mu)
        + target * torch.log(alpha * mu)
        - target * torch.log1p(alpha * mu)
    )

    return -log_prob.mean()


def get_criterion(loss_func: str):
    if args.loss_func == "mse":
        criterion = nn.MSELoss()
    elif args.loss_func == "smape":
        criterion = SMAPE()
    elif args.loss_func == "prob":
        criterion = studentT_nll
    elif args.loss_func == "negative_binomial":
        criterion = negative_binomial_nll
    return criterion


def train_eval(args, config, iteration):
    """
    Runs training and evaluation procedures. If --load_checkpoint flag is
    passed, then trained model is loaded and evaluated

    Args:
        args: Command line arguments
        config: Configuration object for model
        iteration: Current iteration out of args.itr iterations
    """
    model = get_model(architecture=args.model)
    train_loader, val_loader, test_loader = prepare_data_loaders(args, config)
    criterion = get_criterion(loss_func=args.loss_func)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=1e-8)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    trainer = Trainer(
        args,
        config,
        iteration,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        early_stopping,
    )

    if args.load_checkpoint:
        checkpoint_path = get_checkpoint_path(args.loss_func)
        print_rank_0(f"\nLoading model from {checkpoint_path}...")
        trainer.load_checkpoint(checkpoint_path)
    else:
        print_rank_0("\nStarting training procedure...")
        trainer.train()

    return trainer.test()


def main(args):
    if not torch.cuda.is_available():
        print(
            "Could not find GPU. At least one GPU is needed to run this script."
            "\nTerminating now..."
        )
        return

    ddp_setup()

    config = OmegaConf.load(args.config_path)

    if not os.path.exists(args.snapshot_directory):
        os.makedirs(args.snapshot_directory)

    metric_1 = "Average MAE" if args.loss_func == "mse" else "CRPS Sum"
    metric_2 = "Average MSE" if args.loss_func == "mse" else "CRPS"

    start_time = time.time()

    for i in range(args.itr):
        iteration_string = f"Iteration {i + 1}/{args.itr}"
        print_rank_0(f"\n========== {iteration_string} ==========")

        """
        If loss function is set to "mse", then (value_1, value_2) will be
        (average_mae, average_mse).

        Else, (value_1, value_2) will be (crps_sum, crps) 
        """
        value_1, value_2 = train_eval(args, config, i)
        print_rank_0(f"{iteration_string} {metric_1}: {value_1:.4f}")
        print_rank_0(f"{iteration_string} {metric_2}: {value_2:.4f}")

        if args.get_checkpoint_path or args.read_values:
            break

    time_elapsed_min = np.abs((time.time() - start_time) / 60)
    print_rank_0(f"\nFinished! Time elapsed: {time_elapsed_min:.0f} minutes\n")

    destroy_process_group()


"""
Probabilstic forecasting script:
bash ./scripts/monash_prob_demo.sh

Deterministic forecasting script:
bash ./scripts/monash_demo.sh

Parallel probabilstic forecasting script:
bash ./scripts/monash_prob_demo_parallel.sh

Parallel deterministic forecasting script:
bash ./scripts/monash_demo_parallel.sh

! I think something's wrong with the version of PyTorch in the Conda env bc
! I can't use any GPUs with it
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains and evaluates model for time series forecasting"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="weather_GTP4TS_multi-debug",
    )

    checkpoints_directory = "checkpoints"
    if not os.path.exists(checkpoints_directory):
        os.makedirs(checkpoints_directory)

    parser.add_argument(
        "--checkpoints",
        type=str,
        default=checkpoints_directory,
        help="Path to `checkpoints` directory",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        choices=["long_term_forecast"],
        default="long_term_forecast",
        help="Task the model will be trained and evaluated for",
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Total number of time steps in ground truth time series",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=96,
        help="Number of time steps to compute predictions for",
    )
    parser.add_argument(
        "--label_len",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--decay_fac",
        type=float,
        default=0.9,
        help="",  # ? decay factor for learning rate?
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate to use during training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use during training",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="Number of epochs to use during training",
    )
    parser.add_argument(
        "--lradj",
        type=str,
        default="type3",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="",  # number of times to try and get a lower val loss before prematurely ending training
    )
    parser.add_argument(
        "--gpt_layers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--is_gpt",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--e_layers",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Probability of dropout in range [0.0, 1.0]",
    )
    parser.add_argument(
        "--enc_in",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--c_out",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--loss_func",
        type=str,
        choices=["mse", "prob", "negative_binomial"],
        default="mse",
        help='Loss function to minimize during training. Set to "mse" for'
        "deterministic forecasting",
    )
    parser.add_argument(
        "--pretrain",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["DLinear", "TEMPO", "T5", "ETSformer"],
        default="TEMPO",
        help="Model architecture",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--hid_dim",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--tmax",
        type=int,
        default=10,
        help="Max number of iterations over which learning rate will decrease",
    )
    parser.add_argument(
        "--itr",
        type=int,
        default=1,
        help="Number of iterations to run training and evaluation loop",
    )
    parser.add_argument(
        "--cos",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--equal",
        type=int,
        default=1,
        help="1: equal sampling. 0: don't do equal sampling",
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help="whether use prompt pool",
    )
    parser.add_argument(
        "--no_stl_loss",
        action="store_true",
        help="whether use prompt pool",
    )
    parser.add_argument(
        "--stl_weight",
        type=float,
        default=0.01,
    )

    configs_directory = "configs"
    tempo_config = "run_TEMPO.yml"
    tempo_config_path = os.path.join(configs_directory, tempo_config)

    parser.add_argument(
        "--config_path",
        type=str,
        default=tempo_config_path,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="exchange",
        help="Dataset(s) to use during training",
    )
    parser.add_argument(
        "--target_data",
        type=str,
        default="ETTm1",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="exchange",
        help="Dataset(s) to use during evaluation",
    )
    parser.add_argument(
        "--use_token",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--electri_multiplier",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--traffic_multiplier",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of samples to use when computing probabilistic forecasts",
    )
    parser.add_argument(
        "--print_args_config",
        action="store_true",
        help="If passed, will print command line arguments and configuration",
    )
    parser.add_argument(
        "--load_checkpoint",
        action="store_true",
        help="If passed, will load trained model and evaluate it. Deterministic"
        ' model will be loaded if --loss_func is set to "mse". Otherwise,'
        " probabilstic model will be loaded",
    )
    parser.add_argument(
        "--read_values",
        action="store_true",
        help="If passed, will evaluate predicted and true values from .csv file."
        " Determinsitic model's predicted values will be loaded if --loss_func"
        ' is set to "mse". Otherwise, probabilistic model\'s predicted values'
        " will be loaded",
    )
    parser.add_argument(
        "--snapshot_directory",
        type=str,
        default="snapshots",
        help="Directory to save snapshots during training",
    )
    args = parser.parse_args()
    main(args)
