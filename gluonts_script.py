import argparse
import logging
import os

import pytorch_lightning as pl
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.repository import get_dataset
from gluonts.torch.batchify import batchify
from omegaconf import OmegaConf

from lightning_TEMPO import LightningTEMPO
from tempo.utils.data import prepare_data

# Configure logger
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


def main(args):
    # Load model configuration
    data_config = OmegaConf.load("./configs/multiple_datasets.yml")
    logging.debug("Loaded config")

    # Load dataloaders
    (
        _,  # train_data
        train_loader,
        _,  # val_data
        val_loader,
        _,  # test_data
        test_loader,
    ) = prepare_data(args, data_config)

    model_config = OmegaConf.load("./configs/run_TEMPO.yml")
    logging.debug("Loaded config")

    # Initialize TEMPO model
    model = LightningTEMPO(args, model_config)
    logging.debug("Loaded model")

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=args.train_epochs)
    logging.debug("Loaded trainer")

    # Train model
    trainer.fit(model, train_loader)

    logging.debug("Trained model")


"""
Probabilstic forecasting script:
bash ./scripts/monash_prob_demo.sh

Deterministic forecasting script:
bash ./scripts/monash_demo.sh

Parallel probabilstic forecasting script:
bash ./scripts/monash_prob_demo_parallel.sh

Parallel deterministic forecasting script:
bash ./scripts/monash_demo_parallel.sh
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains and evaluates TEMPO model for time series forecasting"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="weather_GTP4TS_multi-debug",
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
        help="",  # ? learning rate adjustment?
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of attempts to get a lower validation loss before"
        "prematurely ending training",
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
        default="prob",
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
    args = parser.parse_args()
    main(args)
