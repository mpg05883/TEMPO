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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
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

# Directory where checkpoints are stored
CHECKPOINTS_DIRECTORY = "checkpoints"

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


def print_rank_0(message: str):
    """
    Prints message on rank 0 process
    """
    if dist.get_rank() == 0:
        print_rank_0(message)


def print_args_config(args, config):
    print_rank_0("\n========== Command line arguments ==========")
    for key, value in vars(args).items():
        print_rank_0(f"{key}: {value}")
    print_rank_0(f"\n========== Config ==========\n{OmegaConf.to_yaml(config)}")


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
    Prepare train, validation and test data loaders.

    Args:
        args: Arguments containing dataset configurations
        config: Configuration dictionary

    Returns:
        tuple: (train_data, train_loader, test_data, test_loader, val_data,
                val_loader)
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


def print_epoch_loss(train_loss, vali_loss):
    """
    Prints two strings. The first string displays current epoch number and
    total number of training steps. Second string displays current epoch's
    training loss and validation loss

    Args:
        train_loss: Model's loss on training set
        vali_loss: Model's loss on validation set

    Returns:
        str: string displaying current epoch number, total number of training
             steps in training set data loader, current epoch's training loss,
             and current epoch's validation loss
    """
    print_rank_0(f"Train loss: {train_loss:.3f} | Val loss: {vali_loss:.3f}")


def print_epoch_time(epoch_start_time):
    """
    Prints string displaying number of minutes elapsed during current epoch

    Args:
        epoch_start_time: Time when current epoch started
    """
    time_elapsed_min = np.abs((time.time() - epoch_start_time) / 60)
    print_rank_0(f"Time elapsed: {time_elapsed_min:.0f} minutes")


def print_batch_updates(
    batch,
    epoch,
    loss,
    batch_start_time,
    num_batches,
    training_steps,
):
    """
    Prints two strings. The first string displays the current batch, epoch,
    and loss. The second string displays the average epoch speed in seconds per
    epoch and estimated remaining time in seconds

    Args:
        batch: Current batch number
        epoch: Current epoch number
        loss: Training loss during current batch
        batch_start_time: Current batch's start time
        num_batches: Total number of batches in current epoch
        training_steps: Total number of training steps in training set data loader
    """
    # Create message displaying current batch, epoch, and loss
    batch_msg = f"Batch: {batch}, epoch: {epoch + 1}, loss: {loss.item():.3f}"

    print_rank_0(batch_msg)

    # Compute average time elapsed per epoch
    speed = (time.time() - batch_start_time) / num_batches
    speed_msg = f"Average epoch speed: {speed:.3f}s/epoch"

    # Estimate remaining amount of time in seconds
    remaining_time_seconds = speed * (
        (args.train_epochs - epoch) * training_steps - batch
    )
    time_msg = f"estiamted remaining time: {remaining_time_seconds:.0f}s"

    # Create message displaying epoch speed and remaining time
    speed_and_time_msg = f"\t{speed_msg}, {time_msg}"

    print_rank_0(speed_and_time_msg)


def get_checkpoint(loss_func: str):
    """
    Returns a file path to a trained model's checkpoint

    Args:
        loss_func: Loss function that model was trained to optimized.
                   Determines if deterministic or probabilistic model
                   checkpoint is returned

    Returns:
        str: File path to trained model's checkpoint
    """
    # Path to checkpoints_directory (./checkpoints)
    checkpoints_directory_path = os.path.join(CHECKPOINTS_DIRECTORY, "Monash_1")

    """
    If loss func is "mse", then get deterministic model's directory. Else, get
    probabilstic model's directory
    """
    is_prob = "_Prob_" if loss_func == "mse" else "_"

    # Directory where model's checkpoint is stored
    model_dir = f"Demo_Monash_TEMPO{is_prob}6_prompt_learn_336_96_100_sl336_ll0_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0"

    # Path to model_dir (./checkpoints/model_dir)
    model_dir_path = os.path.join(checkpoints_directory_path, model_dir)

    # Name of checkpoint file in model_dir
    checkpoint_file = "checkpoint.pth"

    # Path to checkpoint.pth (./checkpoints/model_dir/checkpoint_file)
    checkpoint_path = os.path.join(model_dir_path, checkpoint_file)

    return checkpoint_path


def get_model(args, architecture: str = None):
    if args.model == "PatchTST":
        model = PatchTST(args)
    elif args.model == "DLinear":
        model = DLinear(args)
    elif args.model == "TEMPO":
        model = TEMPO(args)
    elif args.model == "T5":
        model = T54TS(args)
    elif "ETSformer" in args.model:
        model = ETSformer(args)
    else:
        model = GPT4TS(args)
    return model


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


def train(model, args, rank, train_loader, vali_data, vali_loader, iteration):
    """
    Trains a model for either deterministic or probabilstic time series
    forecasting, depending on the loss function specified in args

    Args:
        model: Model to be trained
        args: Command line arguments
        rank: Unique identifier of the GPU that's running
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
    print_rank_0(f"Model will be saved to {model_path}")

    # Specify loss function
    criterion = get_criterion(loss_func=args.loss_func)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.tmax,
        eta_min=1e-8,
    )

    # Get number of training steps
    # training_steps = len(train_loader)

    for epoch in range(args.train_epochs):
        # epoch_start_time = time.time()
        print_rank_0(f"\n========== Epoch {epoch + 1}/{args.train_epochs} ==========")
        # num_batches = 0
        train_loss = []
        train_loader.sampler.set_epoch(epoch)
        vali_loader.sampler.set_epoch(epoch)

        is_not_master_process = dist.get_rank() != 0
        for i, data in tqdm(
            enumerate(train_loader),
            # total=training_steps,
            disable=is_not_master_process,
        ):
            # batch_start_time = time.time()
            # num_batches += 1

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

            batch_x = batch_x.float().to(rank)
            batch_y = batch_y.float().to(rank)
            batch_x_mark = batch_x_mark.float().to(rank)
            batch_y_mark = batch_y_mark.float().to(rank)

            seq_trend = seq_trend.float().to(rank)
            seq_seasonal = seq_seasonal.float().to(rank)
            seq_resid = seq_resid.float().to(rank)

            # Clear gradients
            optimizer.zero_grad()

            # Compute forward pass
            # TODO: restore model files to commit "updated cmd line args" so forward passes use gpu
            if args.model == "TEMPO" or "multi" in args.model:
                outputs, loss_local = model(
                    batch_x,
                    iteration,
                    seq_trend,
                    seq_seasonal,
                    seq_resid,
                )
            elif "former" in args.model:
                dec_inp = (
                    torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(rank)
                )

                dec_inp = (
                    torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(rank)
                )

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = model(batch_x, iteration)

            # Compute current batch's loss
            if args.loss_func == "prob" or args.loss_func == "negative_binomial":
                batch_y = batch_y[:, -args.pred_len :, :].to(rank).squeeze()
                loss = criterion(batch_y, outputs)
            else:
                outputs = outputs[:, -args.pred_len :, :]
                batch_y = batch_y[:, -args.pred_len :, :].to(rank)
                loss = criterion(outputs, batch_y)

            if args.model == "GPT4TS_multi" or args.model == "TEMPO_t5":
                if not args.no_stl_loss:
                    loss += args.stl_weight * loss_local

            train_loss.append(loss.item())

            # Print update every 1000 batches
            if (i + 1) % 1000 == 0:
                pass
                # print_batch_updates(
                #     i,
                #     epoch,
                #     loss,
                #     batch_start_time,
                #     num_batches,
                #     training_steps,
                # )

            # Compute backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

        # Outside for loop

        # Print update after finishing current epoch
        # print_epoch_time(epoch_start_time)

        # Compute current epoch's average training loss
        train_loss = torch.tensor(
            train_loss,
            dtype=torch.float32,
            device=f'cuda"{rank}',
        )

        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)

        train_loss /= dist.get_world_size()

        train_loss = train_loss.item()

        # TODO: paralleize validation loss calculation
        # Compute current epoch's validation loss
        vali_loss = vali(
            model,
            vali_data,
            vali_loader,
            criterion,
            args,
            iteration,
        )

        # Print current epoch's training and validation loss
        print_epoch_loss(train_loss, vali_loss)

        if args.cos:
            scheduler.step()
            print_rank_0(
                "Learning rate: {:.3e}".format(optimizer.param_groups[0]["lr"])
            )
        else:
            adjust_learning_rate(optimizer, epoch + 1, args)

        early_stopping(vali_loss, model, model_path)
        if early_stopping.early_stop:
            print_rank_0(
                f"\nEarlyStopping reached{early_stopping.counter}/{early_stopping.patience}"
            )
            print_rank_0("Ending training procedure early...")
            break

    return model


def train_eval(args, config, rank):
    """
    Runs training and evlauation loop for args.itr iterations

    Args:
        args: Command line arguments
        config: Configuration object for model
        rank: Unique identifier of each process
    """

    for i in range(args.itr):
        print_rank_0(f"\n========== Iteration {i + 1}/{args.itr} ==========")

        # Get data loaders for training, validation, and test sets
        (
            _,  # train_data
            train_loader,
            _,  # test_data
            test_loader,
            vali_data,
            vali_loader,
        ) = prepare_data_loaders(args, config)

        # Initialize model
        model = get_model(args, args.model)

        # Move model to GPU
        model = model.to(rank)

        # Wrap model in DDP
        model = DDP(model, device_ids=[rank])

        # Add attribute for process rank
        model.rank = rank

        if args.get_checkpoint:
            # Get trained model's checkpoint
            checkpoint_path = get_checkpoint(args.loss_func)
            print_rank_0(f"\nLoading model from {checkpoint_path}...")

            # Unwrap model from DDP and load checkpoint
            checkpoint = torch.load(checkpoint_path)
            model.module.load_state_dict(checkpoint, strict=False)
        else:
            print_rank_0("\nStarting training procedure...")
            model = train(
                model,
                args,
                rank,
                train_loader,
                vali_data,
                vali_loader,
                i,
            )

        # TODO: parallelize evaluation code once you've parallelized training code
        print_rank_0("\n========== Evaluating Model ==========")
        """
        If loss function is set to "mse", then (value_1, value_2) will be
        (average_mae, average_mse).

        Else, (value_1, value_2) will be (crps_sum, crps) 
        """
        value_1, value_2 = test(
            model,
            test_loader,
            args,
            # device,
            i,
            read_values=args.read_values,
        )

        metric_1 = "Average MAE" if args.loss_func == "mse" else "CRPS Sum"
        metric_2 = "Average MSE" if args.loss_func == "mse" else "CRPS"

        print_rank_0(f"{metric_1}: {value_1:.4f}")
        print_rank_0(f"{metric_2}: {value_2:.4f}")

        if args.get_checkpoint or args.read_values:
            break


def main(args):
    start_time = time.time()

    ddp_setup()

    config = get_init_config(args.config_path)

    if args.print_args_config:
        print_args_config(args, config)

    rank = int(os.environ["LOCAL_RANK"])
    train_eval(args, config, rank)

    destroy_process_group()

    time_elapsed_min = np.abs((time.time() - start_time) / 60)
    print_rank_0(f"\nFinished! Time elapsed: {time_elapsed_min:.0f} minutes\n")


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

    if not os.path.exists(CHECKPOINTS_DIRECTORY):
        os.makedirs(CHECKPOINTS_DIRECTORY)

    parser.add_argument(
        "--checkpoints",
        type=str,
        default=CHECKPOINTS_DIRECTORY,
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
        "--get_checkpoint",
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
    # # old distribute arguments
    # parser.add_argument(
    #     "--num_workers",
    #     type=int,
    #     default=0,
    #     help="",  # ? number of subprocesses to use for data loading?
    # )
    # parser.add_argument(
    #     "--num_nodes",
    #     type=int,
    #     default=1,
    #     help="",  # ? number of machines to use?
    # )
    args = parser.parse_args()
    main(args)
