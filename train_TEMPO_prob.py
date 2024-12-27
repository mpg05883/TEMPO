import argparse
import logging
import os
import random
import sys
import time
import warnings

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from numpy.random import choice
from omegaconf import OmegaConf

# from torch import optim
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
from tempo.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    test_probs,
    vali,
    visual,
)

warnings.filterwarnings("ignore")

FIX_SEED = 2021
random.seed(FIX_SEED)
torch.manual_seed(FIX_SEED)
np.random.seed(FIX_SEED)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(message)s", datefmt="%m/%d/%Y %I:%M:%S%p"
)

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
    config = OmegaConf.load(config_path)
    return config


def print_config(config):
    print(f"\n\n=== Config ===\n{OmegaConf.to_yaml(config)}")


def print_args(args):
    print("=== Command line arguments ===")
    for key, value in vars(args).items():
        print(f"{key}: {value}")


def get_settings(args, itr, sl=336):
    return "{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}".format(
        args.model_id,
        sl,
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
    print(f"\n\n=== {name} Information ===")
    print(f"Number of samples: {len(data)}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Number of batches: {len(loader)}")

    attributes = ["features", "targets", "shape"]
    for attr in attributes:
        if hasattr(data, attr):
            print(f"{attr}: {getattr(data, attr)}")

    # for batch in loader:
    #     if isinstance(batch, (tuple, list)):
    #         print("\nFirst batch shapes:")
    #         for i, item in enumerate(batch):
    #             print(f"Item {i} shape: {item.shape if hasattr(item, 'shape') else 'N/A'}")
    #     else:
    #         print(f"\nFirst batch shape: {batch.shape if hasattr(batch, 'shape') else 'N/A'}")
    #     break


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
    i,
    epoch,
    loss,
    start_of_batch_time,
    epoch_counter,
    train_steps,
):
    # Print current mini-batch, epoch, and loss
    print(f"Batch: {i}, epoch: {epoch + 1}, loss: {loss.item():.7f}")

    # Compute average time per epoch
    speed = (time.time() - start_of_batch_time) / epoch_counter

    # Compute remaining amount of time in seconds
    remaining_time_seconds = speed * ((args.train_epochs - epoch) * train_steps - i)

    print(
        f"\tAverage epoch speed: {speed:.4f}s/epoch, remaining time: {remaining_time_seconds:.4f}s"
    )


def main(args):
    # Load configuration
    config = get_init_config(args.config_path)

    # Print configuration
    print_config(config)

    # Print command line arguments
    print_args(args)

    for itr in range(args.itr):
        print(f"\n========== Iteration {itr + 1}/{args.itr} ==========")

        if not args.load_finetuned_model:
            # Get name of model's directory
            settings = get_settings(args, itr)

            # Create path to model's directory
            model_path = os.path.join(args.checkpoints, settings)

            # Create model's directory if it doesn't exist
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            print(f"Model will be saved to {model_path}")

        # if args.freq == 0:
        #     args.freq = 'h'

        # Set device to run model on
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model will run on {device}")

        # Load training, validation, and test sets
        (
            train_data,
            train_loader,
            test_data,
            test_loader,
            vali_data,
            vali_loader,
        ) = prepare_data_loaders(args, config)

        # Get number of training steps
        train_steps = len(train_loader)

        # Load model
        if args.load_finetuned_model:
            # Initialize TEMPO model
            model = TEMPO(args, device)

            # Load finetuned model's parameters
            model.load_state_dict(
                torch.load(args.finetuned_model_checkpoint),
                strict=False,
            )
        else:
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

        # Move model to specified device
        model.to(device)

        # Get number of parameters model has
        params = sum(p.numel() for p in model.parameters())
        print(f"Model has {params:,} parameters")

        # If args.load_finetuned_model is set to false, then train model
        if not args.load_finetuned_model:
            print("\n========== Training model ==========")
            # Initialize optimizer
            model_optim = torch.optim.Adam(params, lr=args.learning_rate)

            # Initialize early stopping
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)

            # Set loss function
            if args.loss_func == "mse":
                criterion = nn.MSELoss()
            elif args.loss_func == "smape":
                criterion = SMAPE()
            elif args.loss_func == "prob":
                criterion = studentT_nll
            elif args.loss_func == "negative_binomial":
                criterion = negative_binomial_nll

            # Initialize scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model_optim,
                T_max=args.tmax,
                eta_min=1e-8,
            )

            for epoch in range(args.train_epochs):
                start_of_epoch_time = time.time()
                print(f"\n========== Epoch {epoch + 1}/{args.train_epochs} ==========")
                epoch_counter = 0
                train_loss = []

                for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                    start_of_batch_time = time.time()
                    epoch_counter += 1

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
                            itr,
                            seq_trend,
                            seq_seasonal,
                            seq_resid,
                        )
                    elif "former" in args.model:
                        dec_inp = torch.zeros_like(
                            batch_y[:, -args.pred_len :, :]
                        ).float()
                        dec_inp = (
                            torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1)
                            .float()
                            .to(device)
                        )
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = model(batch_x, itr)

                    # Compute current batch's loss
                    if (
                        args.loss_func == "prob"
                        or args.loss_func == "negative_binomial"
                    ):
                        # outputs = outputs[:, -args.pred_len:, :]
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

                    # Print update
                    if (i + 1) % 1000 == 0:
                        print_batch_update(
                            i,
                            epoch,
                            loss,
                            start_of_batch_time,
                            epoch_counter,
                            train_steps,
                        )

                    # Compute backward pass
                    loss.backward()

                    # Update parameters
                    model_optim.step()

                # Print update after going through current batch
                print(
                    f"Epoch: {epoch + 1}, time elapsed: {((time.time() - start_of_epoch_time) / 60):.0f} minutes"
                )

                # Compute average training set loss
                train_loss = np.average(train_loss)

                # Compute validation set loss
                vali_loss = vali(
                    model, vali_data, vali_loader, criterion, args, device, itr
                )

                # Print current epoch's training and validation lkoss
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

        # Compute crps sum and crps
        print("\n------------------------------------")
        print("Computing CRPS SUM and CRPS...")
        # TODO: add flag to call visual()
        crps_sum, crps = test_probs(
            model,
            test_data,
            test_loader,
            args,
            device,
            itr,
            plot=True,
        )
        print(f"CRPS_Sum: {crps_sum:.4f}")
        print(f"CRPS: {crps:.4f}")


"""
Use this command to run script:
bash ./scripts/monash_prob_demo.sh
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains and evaluates proabilistic forecasting model"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="weather_GTP4TS_multi-debug",
    )

    # Get path where model will be saved after training
    checkpoints_dir = "checkpoints"
    checkpoints_subdirs = [name for name in os.listdir(checkpoints_dir)]

    # name of desired model's directory
    checkpoint = checkpoints_subdirs[0]
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
        help="Name of the task that the model will be trained and evaluated on",
    )
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Input sequence's length",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=96,
        help="Predicted sequence's length",
    )
    parser.add_argument(
        "--label_len",
        type=int,
        default=48,
        help="Ground truth sequence's length",
    )
    parser.add_argument("--decay_fac", type=float, default=0.9)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate to use during training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use during training and inference",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="Number of epochs for training",
    )
    parser.add_argument("--lradj", type=str, default="type3")  # for what
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gpt_layers", type=int, default=6)
    parser.add_argument("--is_gpt", type=int, default=1)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=768)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Probability of a neuron being set to zero in range [0.0, 1.0]",
    )
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument(
        "--loss_func",
        type=str,
        choices=["mse", "prob", "negative_binomial"],
        default="prob",
        help="Loss function to minimize during training",
    )
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--freeze", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        choices=["DLinear", "TEMPO", "T5", "ETSformer"],
        default="TEMPO",
        help="Name of model architecture to used",
    )
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument(
        "--hid_dim",
        type=int,
        default=16,
        help="Number of hidden dimensions",
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
        help="Number of iterations to run training and inference loop",
    )
    parser.add_argument("--cos", type=int, default=0)
    parser.add_argument(
        "--equal",
        type=int,
        default=1,
        help="1: equal sampling. 0: don't do equal sampling",
    )
    parser.add_argument("--pool", action="store_true", help="whether use prompt pool")
    parser.add_argument(
        "--no_stl_loss",
        action="store_true",
        help="whether use prompt pool",
    )
    parser.add_argument("--stl_weight", type=float, default=0.01)

    # Get file path to desired configuration
    configs_dir = "configs"
    configs = [name for name in os.listdir(configs_dir)]

    # Name of config file that'll be used
    config = "run_TEMPO.yml"
    config_path = os.path.join(configs_dir, config)
    parser.add_argument(
        "--config_path",
        type=str,
        default=config_path,
        help="Path to configuration file that'll be used",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="exchange",
        help="Dataset(s) to use during training",
    )
    parser.add_argument("--target_data", type=str, default="ETTm1")
    parser.add_argument(
        "--eval_data",
        type=str,
        default="exchange",
        help="Dataset(s) to use during evaluation",
    )
    parser.add_argument("--use_token", type=int, default=0)
    parser.add_argument("--electri_multiplier", type=int, default=1)
    parser.add_argument("--traffic_multiplier", type=int, default=1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of samples to use when computing probabilistic forecasts",
    )
    parser.add_argument(
        "--load_finetuned_model",
        type=bool,
        default=True,
        help="Set to true if you want to load fine-tuned TEMPO model",
    )
    finetuned_model_checkpoint = os.path.join(checkpoints_dir, "Monash_1")
    finetuned_model_checkpoint = os.path.join(
        finetuned_model_checkpoint,
        "Demo_Monash_TEMPO_Prob_6_prompt_learn_336_96_100_sl336_ll0_pl96_dm768_nh4_el3_gl6_df768_ebtimeF_itr0",
    )
    finetuned_model_checkpoint = os.path.join(
        finetuned_model_checkpoint,
        "checkpoint.pth",
    )
    print(f"Fine-tuned model checkpoint: {finetuned_model_checkpoint}")
    parser.add_argument(
        "--finetuned_model_checkpoint", type=str, default=finetuned_model_checkpoint
    )

    args = parser.parse_args()
    main(args)
