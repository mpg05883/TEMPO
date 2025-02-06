import torch
import torch.distributions as dist


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

    return -log_prob.mean()
