import torch.nn as nn
from criterion import negative_binomial_nll, studentT_nll
from smape import SMAPE


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        # f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        f"trainable params: {trainable_params} || all params: {all_param}"
    )


def get_criterion(loss_func: str):
    if loss_func == "mse":
        criterion = nn.MSELoss()
    elif loss_func == "smape":
        criterion = SMAPE()
    elif loss_func == "prob":
        criterion = studentT_nll
    elif loss_func == "negative_binomial":
        criterion = negative_binomial_nll
    return criterion
