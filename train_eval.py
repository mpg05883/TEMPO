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
from torch.utils.data import Subset
from tqdm import tqdm

from smape import SMAPE
from tempo.data_provider.data_factory import data_provider
from tempo.models.DLinear import DLinear
from tempo.models.ETSformer import ETSformer
from tempo.models.GPT4TS import GPT4TS
from tempo.models.PatchTST import PatchTST
from tempo.models.T5 import T54TS
from tempo.models.TEMPO import TEMPO
from tempo.utils.tools import EarlyStopping, adjust_learning_rate, test, vali

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


def print_args_and_config(args, config):
    print("\n========== Command line arguments ==========")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print(f"\n========== Config ==========\n{OmegaConf.to_yaml(config)}")


def get_init_config(config_path=None):
    """
    Retrieves an initial configuration from a specified file path.

    Args:
        config_path (str, optional): Path to a configuration file that'll be
        used to initialize the model. Defaults to None.

    Returns:
        config (OmegaConf): Configuration object for initializing the model
    """
    config = OmegaConf.load(config_path)
    return config


def get_settings(args, itr, seq_len=336):
    return "{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}".format(
        args.model_id,
        seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.gpt_layers,
        args.d_ff,
        args.embed,
        itr,
    )


def print_dataset_info(data, loader, name="Dataset"):
    print(f"\n========== {name} Info ==========")
    print(f"Number of samples: {len(data):,}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Number of batches: {len(loader)}")

    attributes = ["features", "targets", "shape"]
    for attr in attributes:
        if hasattr(data, attr):
            print(f"{attr}: {getattr(data, attr)}")


def prepare_data_loaders(args, config):
    """
    Prepare train, validation and test data loaders.

    Args:
        args: Arguments containing dataset configurations
        config: Configuration dictionary

    Returns:
        tuple: (train_data, train_loader, test_data, test_loader, val_data, val_loader)
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
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    # Prepare test data
    _update_args_from_config(args, config, args.target_data)
    test_data, test_loader = data_provider(args, "test")

    print_dataset_info(train_data, train_loader, "Training Dataset")
    print_dataset_info(val_data, val_loader, "Validation Dataset")
    print_dataset_info(test_data, test_loader, "Test Dataset")

    return train_data, train_loader, test_data, test_loader, val_data, val_loader


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


def _combine_datasets(datasets):
    """Combine multiple datasets into one"""
    combined = datasets[0]
    for dataset in datasets[1:]:
        combined = torch.utils.data.ConcatDataset([combined, dataset])
    return combined


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


def get_epoch_loss_msg(epoch, train_steps, train_loss, vali_loss):
    """
    Creates string displaying current epoch number, total number of training
    steps in training set data loader, current epoch's training loss, and
    current epoch's validation loss

    Args:
        epoch: Current epoch's number
        train_steps: Total number of training steps in training set data loader
        train_loss: Model's loss on training set
        vali_loss: Model's loss on validation set

    Returns:
        str: string displaying current epoch number, total number of training
             steps in training set data loader, current epoch's training loss,
             and current epoch's validation loss
    """
    return f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Val Loss: {vali_loss:.7f}"


def get_epoch_time_msg(epoch, epoch_start_time):
    """
    Creates string displaying current epoch number and number of minutes
    elapsed during current epoch

    Args:
        epoch: Current epoch's number
        epoch_start_time: Time when current epoch started

    Returns:
        str: string displaying current epoch number and number of minutes
             elapsed during current epoch
    """
    elapsed_time_min = (time.time() - epoch_start_time) / 60
    return f"Epoch: {epoch + 1}, time elapsed: {elapsed_time_min:.0f} minutes"


def get_batch_update_msgs(
    batch,
    epoch,
    loss,
    batch_start_time,
    num_batches,
    train_steps,
):
    """
    Generates two strings. The first string displays the current batch, epoch,
    and loss. The second string displays the average epoch speed in seconds per
    epoch and estimated remaining time in seconds

    Args:
        batch: Current batch number
        epoch: Current epoch number
        loss: Training loss during current batch
        batch_start_time: Current batch's start time
        num_batches: Total number of batches in current epoch
        train_steps: Total number of training steps in training set data loader

    Returns:
        tuple: (batch_msg, speed_and_time_msg)
    """
    # Print current batch, epoch, and loss
    batch_msg = f"Batch: {batch}, epoch: {epoch + 1}, loss: {loss.item():.7f}"

    # Compute average time elapsed per epoch
    speed = (time.time() - batch_start_time) / num_batches
    speed_msg = f"Average epoch speed: {speed:.4f}s/epoch"

    # Estimate remaining amount of time in seconds
    remaining_time_seconds = speed * ((args.train_epochs - epoch) * train_steps - batch)
    time_msg = f"estiamted remaining time: {remaining_time_seconds:.4f}s"

    speed_and_time_msg = f"\t{speed_msg}, {time_msg}"

    return batch_msg, speed_and_time_msg


def get_checkpoint(loss_func: str):
    """
    Returns a file path to a trained model's checkpoint

    Args:
        loss_func: Loss function that trained model optimized. Determines
                   if deterministic or probabilistic checkpoint is returned

    Returns:
        str: File path to trained model's checkpoint
    """
    # Checkpoints directory
    checkpoints_dir = "checkpoints"

    # Directory path to checkpoints
    checkpoints_dir_path = os.path.join(checkpoints_dir, "Monash_1")

    if loss_func == "mse":
        # Get deterministic model
        checkpoints_dir_path = os.path.join(
            checkpoints_dir_path,
            "Demo_Monash_TEMPO_6_prompt_learn_336_96_100_sl336_ll0_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0",
        )
    else:
        # Get probabilstic model
        checkpoints_dir_path = os.path.join(
            checkpoints_dir_path,
            "Demo_Monash_TEMPO_Prob_6_prompt_learn_336_96_100_sl336_ll0_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0",
        )

    # File path to checkpoint
    checkpoint_file_path = os.path.join(checkpoints_dir_path, "checkpoint.pth")

    return checkpoint_file_path


def train_model(args, device, train_loader, vali_data, vali_loader, iteration):
    """
    Trains a model for either deterministic or probabilstic time series
    forecasting, depending on the loss function specified in args

    Args:
        args: Command line arguments
        device: Device to run model on
        train_loader: Training set's data loader
        vali_data: Validation set
        vali_loader: Validation set's data loader
        iteration: Iteration number of for loop in main()

    Returns:
        model: Trained model
    """
    # Get name of directory where model will be saved
    settings = get_settings(args, iteration)

    # Create path to directory where model will be saved
    model_path = os.path.join(args.checkpoints, settings)

    # Create directory where model will be saved if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print(f"Model will be saved to {model_path}")

    # Load model
    if args.model == "PatchTST":
        model = PatchTST(args, device)
    elif args.model == "DLinear":
        model = DLinear(args, device)
    elif args.model == "TEMPO":
        model = TEMPO(args, device)
    elif args.model == "T5":
        model = T54TS(args, device)
    elif "ETSformer" in args.model:
        model = ETSformer(args, device)
    else:
        model = GPT4TS(args, device)

    model.to(device)

    # Specify loss function
    if args.loss_func == "mse":
        criterion = nn.MSELoss()
    elif args.loss_func == "smape":
        criterion = SMAPE()
    elif args.loss_func == "prob":
        criterion = studentT_nll
    elif args.loss_func == "negative_binomial":
        criterion = negative_binomial_nll

    # Initialize optimizer
    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optim,
        T_max=args.tmax,
        eta_min=1e-8,
    )

    # Get number of training steps
    train_steps = len(train_loader)

    # Train model for args.train_epochs epochs
    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        print(f"\n========== Epoch {epoch + 1}/{args.train_epochs} ==========")
        num_batches = 0
        train_loss = []

        for i, data in tqdm(enumerate(train_loader), total=train_steps):
            batch_start_time = time.time()
            num_batches += 1

            batch_x, batch_y, batch_x_mark, batch_y_mark = (
                data[0],  # Input time series
                data[1],  # Future time series
                data[2],
                data[3],
            )

            seq_trend, seq_seasonal, seq_resid = (
                data[4],  # Trend component
                data[5],  # Seasonal component
                data[6],  # Residual component
            )

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            seq_trend = seq_trend.float().to(device)
            seq_seasonal = seq_seasonal.float().to(device)
            seq_resid = seq_resid.float().to(device)

            # Clear gradients
            model_optim.zero_grad()

            # Compute forward pass
            if args.model == "TEMPO" or "multi" in args.model:
                outputs, loss_local = model(
                    batch_x,
                    iteration,
                    seq_trend,
                    seq_seasonal,
                    seq_resid,
                )
            elif "former" in args.model:
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()

                dec_inp = (
                    torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(device)
                )

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, iteration)

            # Compute current batch's loss
            if args.loss_func == "prob" or args.loss_func == "negative_binomial":
                batch_y = batch_y[:, -args.pred_len :, :].to(device).squeeze()
                loss = criterion(batch_y, outputs)
            else:
                outputs = outputs[:, -args.pred_len :, :]
                batch_y = batch_y[:, -args.pred_len :, :].to(device)
                loss = criterion(outputs, batch_y)

            if args.model == "GPT4TS_multi" or args.model == "TEMPO_t5":
                if not args.no_stl_loss:
                    loss += args.stl_weight * loss_local

            train_loss.append(loss.item())

            # Print update every 1000 batches
            if (i + 1) % 1000 == 0:
                batch_msg, speed_and_time_msg = get_batch_update_msgs(
                    i,
                    epoch,
                    loss,
                    batch_start_time,
                    num_batches,
                    train_steps,
                )

                print(batch_msg)

                print(speed_and_time_msg)

            # Compute backward pass
            loss.backward()

            # Update parameters
            model_optim.step()

        # Print update after finishing current epoch
        epoch_time_msg = get_epoch_time_msg(epoch, epoch_start_time)
        print(epoch_time_msg)

        # Compute current epoch's average training loss
        train_loss = np.average(train_loss)

        # Compute current epoch's validation loss
        vali_loss = vali(
            model,
            vali_data,
            vali_loader,
            criterion,
            args,
            device,
            iteration,
        )

        # Print current epoch's training and validation loss
        epoch_loss_msg = get_epoch_loss_msg(epoch, train_steps, train_loss, vali_loss)
        print(epoch_loss_msg)

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]["lr"]))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)

        early_stopping(vali_loss, model, model_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model


# TODO: parallelize training and evaluation script
def main(args):
    # Load configuration
    config = get_init_config(args.config_path)

    if args.print_args_and_config:
        print_args_and_config(args, config)

    # Specify device to run model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nModel will run on {device}")

    for i in range(args.itr):
        print(f"\n========== Iteration {i + 1}/{args.itr} ==========")

        # Get data loaders for training, validation, and test sets
        (
            _,  # train_data
            train_loader,
            _,  # test_data
            test_loader,
            vali_data,
            vali_loader,
        ) = prepare_data_loaders(args, config)

        if args.get_checkpoint:
            # Initialize TEMPO model
            model = TEMPO(args, device)

            # Get trained model's checkpoint
            checkpoint_file_path = get_checkpoint(args.loss_func)

            print(f"\nLoading trained model from {checkpoint_file_path}...")

            # Load trained model's parameters
            model.load_state_dict(torch.load(checkpoint_file_path), strict=False)
        else:
            print("\nStarting training procedure...")
            model = train_model(args, device, train_loader, vali_data, vali_loader, i)

        print("\n========== Evaluating Model ==========")
        """
        If loss function is set to "mse", then (value_1, value_2) will be
        (average_mae, average_mse).
        
        Else, (value_1, value_2) will be (crps_sum, crps) 
        """
        value_1, value_2 = test(
            model,
            test_loader,
            args,
            device,
            i,
            read_values=args.read_values,
        )

        metric_1 = "Average MAE" if args.loss_func == "mse" else "CRPS Sum"
        metric_2 = "Average MSE" if args.loss_func == "mse" else "CRPS"

        print(f"{metric_1}: {value_1:.4f}")
        print(f"{metric_2}: {value_2:.4f}")

    # Outside of for loop
    print("\nFinished!\n")


"""
To run probabilstic forecasting script, use:
bash ./scripts/monash_prob_demo.sh

To run deterministic forecasting script, use:
bash ./scripts/monash_demo.sh
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains and evaluates model on time series forecasting"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="weather_GTP4TS_multi-debug",
    )

    # Name of checkpoints directory
    checkpoints_dir = "checkpoints"

    # Create checkpoints directory if it doesn't exist
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Create default directory to save model
    checkpoint = "default"

    # Create path to default directory
    checkpoints_path = os.path.join(checkpoints_dir, checkpoint)

    parser.add_argument(
        "--checkpoints",
        type=str,
        default=checkpoints_path,
        help="Directory path where model will be saved",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        choices=["long_term_forecast"],
        default="long_term_forecast",
        help="Task that model will be trained and evaluated on",
    )
    parser.add_argument(
        "--prompt",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
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
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate to use during training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use during training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="Number of epochs to use during traiing",
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
        help="Loss function to minimize during training",
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
        help="Model architecture that'll be trained and evaluated",
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

    # Name of configurations directory
    configs_dir = "configs"

    # Name of configuration
    config = "run_TEMPO.yml"

    # Create file path to configuration
    config_path = os.path.join(configs_dir, config)

    parser.add_argument(
        "--config_path",
        type=str,
        default=config_path,
        help="Path to configuration file to use during training and evaluation",
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
        "--print_args_and_config",
        type=bool,
        default=False,
        help="Set to True to print command line arguments and configuration",
    )
    parser.add_argument(
        "--get_checkpoint",
        type=bool,
        default=True,
        help="Set to True to skip training procedure and evaluate trained model",
    )
    parser.add_argument(
        "--read_values",
        type=bool,
        default=True,
        help="Set to True to read predicted and true values from .csv file",
    )

    args = parser.parse_args()
    main(args)
