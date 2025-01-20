import os

import numpy as np
import torch
from torch.distributed import all_gather, get_world_size


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true + 1e-8)))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + 1e-8)))


def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
    # return np.mean(200 * np.abs(pred - true) / (pred + true + 1e-8))


def ND(pred, true):
    return np.mean(np.abs(true - pred)) / np.mean(np.abs(true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    nd = ND(pred, true)

    return mae, mse, rmse, mape, mspe, smape, nd


def aggregate_tensors(tensor: torch.Tensor):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = get_world_size()
    tensor_list = [
        torch.zeros_like(tensor, device=local_rank) for _ in range(world_size)
    ]
    all_gather(tensor_list, tensor)
    stacked_tensor = torch.stack(tensor_list)
    return stacked_tensor


def aggregate_metrics(total, num_samples):
    """
    Takes the average of a metric, then aggregates the averages computed by all
    processes.

    Args:
        total: Total accumulated metric
        num_samples: Number of samples
    """
    # Compute this process's average
    local_average = total / num_samples
    local_rank = int(os.environ["LOCAL_RANK"])
    local_average_tensor = torch.tensor(
        local_average,
        dtype=torch.float32,
        device=local_rank,
    )

    # Aggregate average tensors across all processes
    stacked_tensor = aggregate_tensors(local_average_tensor)

    # Get aggregated average
    aggregated_average = torch.mean(stacked_tensor)

    return aggregated_average
