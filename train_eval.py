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
    print(f"\n=== {name} Information ===")
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
        _type_: Negative log-likelihood
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
        _type_: Negative log-likelihood
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


def print_batch_update(
    batch,
    epoch,
    loss,
    batch_start_time,
    num_epochs,
    train_steps,
):
    # Print current batch, epoch, and loss
    print(f"Batch: {batch}, epoch: {epoch + 1}, loss: {loss.item():.7f}")

    # Compute average time per epoch
    speed = (time.time() - batch_start_time) / num_epochs

    # Estimate remaining amount of time in seconds
    remaining_time_seconds = speed * ((args.train_epochs - epoch) * train_steps - batch)

    print(
        f"\tAverage epoch speed: {speed:.4f}s/epoch, remaining time: {remaining_time_seconds:.4f}s"
    )


def load_trained_model(args, device: str, loss_func="mse"):
    """
    Initializes TEMPO model and loads trained model's parameters

    Args:
        loss_func: Loss function that trained model optimized. Determines
                   if deterministic or probabilistic model is loaded
    """
    # Initialize TEMPO model
    model = TEMPO(args, device)

    checkpoints_dir = "checkpoints"

    # Directory path to checkpoints
    checkpoint = os.path.join(checkpoints_dir, "Monash_1")

    if loss_func == "mse":
        # Get deterministic model
        checkpoint = os.path.join(
            checkpoint,
            "Demo_Monash_TEMPO_Prob_6_prompt_learn_336_96_100_sl336_ll0_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0",
        )
    else:
        # Get probabilstic model
        checkpoint = os.path.join(
            checkpoint,
            "Demo_Monash_TEMPO_Prob_6_prompt_learn_336_96_100_sl336_ll0_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0",
        )

    # File path to checkpoint
    checkpoint = os.path.join(checkpoint, "checkpoint.pth")

    # Load trained model's parameters
    return model.load_state_dict(torch.load(checkpoint), strict=False)


def train_model(args, device, train_loader, vali_data, vali_loader, iteration):
    """
    Trains a model for either deterministic or probabilstic time series
    forecasting

    Args:
        args: Command line arguments
        device: Device to run model on
        train_loader: Data loader for training set
        vali_data: Validation set
        vali_loader: Data loader for validation set
        iteration: Iteration number

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

    # Set loss function
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

    for epoch in range(args.train_epochs):
        epoch_start_time = time.time()
        print(f"\n========== Epoch {epoch + 1}/{args.train_epochs} ==========")
        num_epochs = 0
        train_loss = []

        for i, data in tqdm(enumerate(train_loader), total=train_steps):
            batch_start_time = time.time()
            num_epochs += 1

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

            # Print an update every 1000 batches
            if (i + 1) % 1000 == 0:
                print_batch_update(
                    i,
                    epoch,
                    loss,
                    batch_start_time,
                    num_epochs,
                    train_steps,
                )

            # Compute backward pass
            loss.backward()

            # Update parameters
            model_optim.step()

        # Print update after finishing current epoch
        print(
            f"Epoch: {epoch + 1}, time elapsed: {((time.time() - epoch_start_time) / 60):.0f} minutes"
        )

        # Compute current epoch's average training loss
        train_loss = np.average(train_loss)

        # Compute current epoch's validation loss
        vali_loss = vali(
            model, vali_data, vali_loader, criterion, args, device, iteration
        )

        # Print current epoch's training and validation loss
        print(
            f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Val Loss: {vali_loss:.7f}"
        )

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


# TODO: once you finish merging det and prob scripts, ask Defu what notebook he used to start parallelizing the training script
def main(args):
    # Load configuration
    config = get_init_config(args.config_path)

    if args.print_args_and_config:
        print("\n========== Command line arguments ==========")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
        print(f"\n========== Config ==========\n{OmegaConf.to_yaml(config)}")

    # Set device to run model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model will run on {device}")

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

        if args.load_trained_model:
            print("\nLoading trained model...")
            model = load_trained_model(args, device, args.loss_func)
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

        if args.loss_func == "mse":
            print(f"Average MAE: {value_1:.4f}")
            print(f"Average MSE: {value_2:.4f}")
        else:
            print(f"CRPS Sum: {value_1:.4f}")
            print(f"CRPS: {value_2:.4f}")


"""
Probabilstic forecasting script:
bash ./scripts/monash_prob_demo.sh

Deterministic forecasting script:
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
        help="Number of training epochs",
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
        help="Type of model that'll be trained and evaluated",
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

    # Name of configs directory
    configs_dir = "configs"

    # Name of config
    config = "run_TEMPO.yml"

    # Create file path to config
    config_path = os.path.join(configs_dir, config)

    parser.add_argument(
        "--config_path",
        type=str,
        default=config_path,
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
        "--print_args_and_config",
        type=bool,
        default=False,
        help="Set to True to print command line arguments and configuration",
    )
    parser.add_argument(
        "--load_trained_model",
        type=bool,
        default=False,
        help="Set to True to load trained model",
    )
    parser.add_argument(
        "--read_values",
        type=bool,
        default=False,
        help="Set to True to read predicted and true values from a .csv file",
    )

    args = parser.parse_args()
    main(args)
