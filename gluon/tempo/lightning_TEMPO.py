import lightning.pytorch as pl
import torch
from gluonts.torch import PyTorchPredictor
from gluonts.torch.distributions import StudentTOutput

from tempo.models.TEMPO import TEMPO


class LightningTEMPO(TEMPO, pl.LightningModule):
    def __init__(self, args, config, distr_output=StudentTOutput()):
        super().__init__(args, config)
        # Commmand line arguments
        self.args = args

        # Model configuration
        self.config = config

        # TODO: once you get a prototype working, change the code to allow for different output distributions
        # Type of distribution for model output. Default is Student's t-distribution
        self.distr_output = distr_output

    def training_step(self, batch, batch_index):
        """
        Defines the logic for a single training loop iteration.
        """
        # Past time series values
        past_target = batch["past_target"]

        # Future time series values
        future_target = batch["future_target"]

        # TODO: figure out how to get trend, seasonal, and residual components from GluonTS datasets
        # Compute forward pass to get Student's t-distribution parameters
        distr_args, loss_local = TEMPO.forward(x=past_target)

        # Create Student's t-distribution
        student_t_distr = self.distr_output.distribution(distr_args)

        # TODO: once you get a prototype working, change the code to compute different losses based on output distribution
        # Compute Student's t negative log-likelihood loss
        loss = -student_t_distr.log_prob(future_target)

        return loss.mean()

    # TODO:
    def validation_step(self, batch, batch_index):
        """
        Defines the logic for a single validation loop iteration.
        """
        pass

    # TODO:
    def test_step(self, batch, batch_index):
        """
        Defines a single test iteration.
        """
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def get_predictor(self):
        """
        Returns predictor for performing inference.
        """
        return PyTorchPredictor(
            prediction_length=self.args.pred_len,  # Number of future time steps to predict
            input_names=["past_target"],  # Key in batch where time series values are
            prediction_net=super,  # Model that'll be computing the predictions
            batch_size=self.args.batch_size,  # Number of samples in each batch
        )
