import argparse
import logging
import os
from collections.abc import Iterable

from omegaconf import OmegaConf

from tempo.utils.data import prepare_data

# Configure logger
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


def print_config(config: OmegaConf):
    """
    Prints a yaml dump of a specified configuration
    """
    logging.info(f"\n\n=== Config ===\n{OmegaConf.to_yaml(config)}")


def print_args(args):
    """
    Prints each key-value pair in args
    """
    logging.info("=== Command line arguments ===")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")


def main(args):
    # Load configuration
    config = OmegaConf.load(args.config_path)

    # Print args and config
    if args.print_args_and_config:
        print_args(args)
        print_config(config)

    # Load training, validation, and test sets
    (
        train_data,
        train_loader,
        val_data,
        val_loader,
        test_data,
        test_loader,
    ) = prepare_data(args, config)
    
    print(f'train_data type: {type(train_data)}')
    print(f'train_loader type: {type(train_loader)}')
    
    original_dataset = train_loader.dataset
    print(f'original_dataset type: {type(original_dataset)}')


"""
Use this command to run script:
bash ./scripts/scratch.sh
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dummy Python script for messing around with the datasets"
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
        help="Name of the task that the model will be trained and evaluated on",
    )
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=96,
        help="Number of future time steps to generate predictions for",
    )
    parser.add_argument(
        "--label_len",
        type=int,
        default=48,
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
        help="Batch size to use during training",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
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
        help="Probability of dropout in range [0.0, 1.0]",
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
        help="Loss function for training",
    )
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--freeze", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        choices=["DLinear", "TEMPO", "T5", "ETSformer"],
        default="TEMPO",
        help="Model architecture",
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
        help="Path to configuration file",
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
        "--print_args_and_config",
        type=bool,
        default=True,
        help="Set to true to print the cmd line args and config",
    )
    parser.add_argument(
        "--load_finetuned_model",
        type=bool,
        default=True,
        help="Set to true load fine-tuned TEMPO model",
    )
    parser.add_argument(
        "--read_values",
        type=bool,
        default=True,
        help="Set to True to read predicted and true values from a .csv file",
    )
    parser.add_argument(
        "--values_file",
        type=str,
        default="values.csv",
        help="Name of .csv file where predicted and true will be read from",
    )

    args = parser.parse_args()
    main(args)
