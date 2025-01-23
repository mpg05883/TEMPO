import logging
import os
from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    aggregate_metrics,
    aggregate_tensors,
    calc_quantile_CRPS,
    calc_quantile_CRPS_sum,
    distributed_print,
    get_global_rank,
    get_local_rank,
    is_main_process,
)

# Hide matplotlib warnings
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class Trainer:
    def __init__(
        self,
        args: Namespace,
        config: DictConfig,
        iteration: int,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion,
        optimizer,
        scheduler,
        early_stopping: EarlyStopping,
        snapshots_directory: str = "snapshots",
        from_scratch: bool = True,
    ):
        self.args = args
        self.config = config
        self.iteration = iteration
        self.local_rank = get_local_rank()
        self.global_rank = get_global_rank()
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.epochs_ran = 0  # Number of epochs ran before training was interrupted
        self.snapshots_directory = snapshots_directory
        if not os.path.exists(self.snapshots_directory):
            os.makedirs(self.snapshots_directory, exist_ok=True)
        self.snapshot_path = os.path.join(
            self.snapshots_directory,
            self.args.snapshot_file_name,
        )
        if not from_scratch:
            self._load_snapshot()

    # ========================= Public Methods =========================
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads a trained model's parameters from a given checkpoint

        Args:
            checkpoint_path (str): File path to desired checkpoint
        """
        distributed_print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        self.model.module.load_state_dict(checkpoint, strict=False)

    def train(self) -> None:
        """
        Trains model for time series forecasting. Saves a snapshot of the model
        and training state after each epoch
        """
        # Create directory to save model's checkpoints
        settings = self._get_settings(self.args, self.iteration)
        checkpoints_path = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        distributed_print(f"Model will be saved to ./{checkpoints_path}")

        # Set model to training mode
        self.model.train()
        distributed_print(
            f"Training mode: {self.model.module.training}",
            debug=True,
        )

        # Train model for self.args.train_epochs epochs. If training was
        # interrupted and a previous checkpoint was saved, then resume training
        # from that checkpoint
        for epoch in range(self.epochs_ran, self.args.train_epochs):
            distributed_print(f"\nEpoch {epoch + 1}/{self.args.train_epochs}")

            # Total training loss on a single process
            total_train_loss = 0.0

            # Number of samples seen on a single process
            num_samples = 0

            # Set epoch for distributed sampler
            self.train_loader.sampler.set_epoch(epoch)

            # Only display progress bar on main process
            if is_main_process():
                progress_bar = tqdm(total=len(self.train_loader))

            for data in self.train_loader:
                # Unpack data to get tensors
                (
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    seq_trend,
                    seq_seasonal,
                    seq_resid,
                ) = self._unpack_tensor(data)

                # Run forward pass and compute batch loss
                loss, batch_size = self._run_batch(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    seq_trend,
                    seq_seasonal,
                    seq_resid,
                )

                # Backwards pass
                loss.backward()

                # Update parameters
                self.optimizer.step()

                # Increment total training loss and number of samples
                total_train_loss += loss.item()
                num_samples += batch_size

                if is_main_process():
                    progress_bar.update(1)  # Update progress bar

            # Outside of for loop
            if is_main_process():
                progress_bar.close()  # Close progress bar
                self._save_snapshot(epoch)  # Only save snapshot on main process

            # Print number of samples seen on each process for debugging
            distributed_print(
                f"Saw {num_samples} samples",
                debug=True,
                to_all=True,
                epoch=epoch,
            )

            # Print local training loss for debugging
            local_train_loss = total_train_loss / num_samples
            distributed_print(
                f"Local train loss: {local_train_loss:.3f} = {total_train_loss:.3f}/{num_samples}",
                debug=True,
                to_all=True,
                epoch=epoch,
            )

            # Compute average training loss across all processes for this epoch
            average_train_loss = aggregate_metrics(total_train_loss, num_samples)

            # Compute average validation loss across all processes for this epoch
            average_val_loss = self._evaluate_val_set(epoch)

            # Print current epoch's average training loss and validation loss
            distributed_print(
                f"Average train loss: {average_train_loss:.3f}"
                f" | Average validation loss: {average_val_loss:.3f}",
                epoch=epoch,
            )

            if self.args.cos:
                self.scheduler.step()
                distributed_print(
                    "Learning rate: {:.3e}".format(self.optimizer.param_groups[0]["lr"])
                )
            else:
                adjust_learning_rate(self.optimizer, epoch + 1, self.args)

            self.early_stopping(
                average_val_loss,
                self.model,
                checkpoints_path,
                self.global_rank,
            )
            if self.early_stopping.early_stop:
                distributed_print(
                    f"\nEarlyStopping reached {self.early_stopping.counter}/{self.early_stopping.patience}"
                    "\nEnding training procedure early..."
                )
                return
            if self.early_stopping.early_stop:
                distributed_print(
                    f"\nEarlyStopping reached {self.early_stopping.counter}/{self.early_stopping.patience}"
                    "\nEnding training procedure early..."
                )
                return

    def test(
        self,
        plot=True,
        plot_file_name="pred_vs_true.png",
        overwrite_csv=True,
        csv_file_name="det_values.csv",
        batch_index=0,
        instance_index=0,
    ) -> tuple[float, float]:
        """
        Evaluates model on the test set and measures its performance using
        average MAE and average MSE. If a probabilstic model was trained, then
        performance is measured using CRPS sum and CRPS.

        Args:
            plot (bool, optional): If True, will plot predicted and true values.
                                   Defaults to True.
            plot_file_name (str, optional): File name of the predicted vs true
                                            values plot. Defaults to
                                            "pred_vs_true.png".
            overwrite_csv (bool, optional): If True, will overwrite the values
                                            previously saved predicted and true
                                            values in the .csv file. Defaults
                                            to True.
            csv_file_name (str, optional): File name of the .csv file where
                                           predicted and true values will be
                                           saved. Defaults to "det_values.csv".
            batch_index (int, optional): Index of the batch that'll be plotted
                                         and saved. Defaults to 0.
            instance_index (int, optional): Index of the instance that'll be
                                            plotted and saved. Defaults to 0.

        Returns:
            tuple[float, float]: (average MAE, average MSE) for deterministic
                                 models and (CPRS sum, CRPS) for probabilstic
                                 models
        """
        # If model is probabilistic, then call _test_probs() to compute CRPS sum
        # and CRPs
        if self.args.loss_func != "mse":
            # Keep default csv file name when name calling _test_probs()
            return self._test_probs(
                plot=plot,
                plot_file_name=plot_file_name,
                overwrite_csv=True,
                batch_index=batch_index,
                instance_index=instance_index,
            )

        # If --read_values flag is set to True and this is the main process,
        # then read values from .csv file and create plot before going through
        # the rest of this method to save time
        if self.args.read_values and is_main_process():
            self._read_and_plot(csv_file_name, plot_file_name)

        y_pred = torch.tensor([], device=self.local_rank)  # Predicted values
        y_true = torch.tensor([], device=self.local_rank)  # True values

        # Initialize torchmetrics objects for computing MAE and MSE
        mean_absolute_error = MeanAbsoluteError().to(self.local_rank)
        mean_squared_error = MeanSquaredError().to(self.local_rank)

        total_mae = 0.0  # Total MAE for a single process
        total_mse = 0.0  # Total MSE for a single process
        num_samples = 0  # Number of samples seen on a single process

        # Set model to evaluation mode
        self.model.eval()
        distributed_print(
            f"Evaluation mode: {not self.model.module.training}",
            debug=True,
        )

        with torch.no_grad():
            # Set epoch for distributed dataloader
            self.test_loader.sampler.set_epoch(self.iteration)

            # Only show progress bar on main process
            if is_main_process():
                progress_bar = tqdm(total=len(self.test_loader))

            for data in self.test_loader:
                # Unpack data to get tensors
                (
                    batch_x,
                    batch_y_true,
                    batch_x_mark,
                    batch_y_mark,
                    seq_trend,
                    seq_seasonal,
                    seq_resid,
                ) = self._unpack_tensor(data)

                # Compute forward pass
                batch_y_pred, _ = self._forward_pass(
                    batch_x,
                    batch_y_true,
                    batch_x_mark,
                    batch_y_mark,
                    seq_trend,
                    seq_seasonal,
                    seq_resid,
                )

                # Save last pred_len values
                batch_y_pred = batch_y_pred[:, -self.args.pred_len :, :].detach()
                batch_y_true = batch_y_true[:, -self.args.pred_len :, :].detach()
                y_pred = torch.cat((y_pred, batch_y_pred))
                y_true = torch.cat((y_true, batch_y_true))

                # Compute batch MAE and batch MSE
                batch_mae = mean_absolute_error(batch_y_pred, batch_y_true)
                batch_mse = mean_squared_error(batch_y_pred, batch_y_true)

                # Increment total MAE, total MSE, and number of samples
                batch_size = batch_x.size(0)
                total_mae += batch_mae * batch_size
                total_mse += batch_mse * batch_size
                num_samples += batch_size

                if is_main_process():
                    progress_bar.update(1)  # Update progress bar

        if is_main_process():
            progress_bar.close()  # Close progress bar

        # Aggregate y_true and y_pred from all processes
        aggregated_y_true = aggregate_tensors(y_true)
        aggregated_y_pred = aggregate_tensors(y_pred)

        # If plot flag is set to True, overwrite_csv flag is set to True, and
        # this is the main process, then write aggregated_y_true and
        # aggregated_y_pred to the specified .csv file and create the predicted
        # vs true values plot
        if plot and overwrite_csv and is_main_process():
            self._write_and_plot(
                csv_file_name,
                plot_file_name,
                aggregated_y_true,
                aggregated_y_pred,
                batch_index,
                instance_index,
            )

        # Compute average MAE and MSE loss across all processes
        average_mae = aggregate_metrics(total_mae, num_samples)
        average_mse = aggregate_metrics(total_mse, num_samples)

        return average_mae, average_mse

    # ========================= Private Methods =========================
    # ========== Snapshots ==========
    def _load_snapshot(self) -> None:
        if not os.path.exists(self.snapshot_path):
            distributed_print(
                "Cannot find previous snapshot. Starting training from Epoch 0",
                warning=True,
            )
            return

        map_location = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.snapshot_path, map_location=map_location)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_ran = snapshot["EPOCHS_RAN"]
        distributed_print(f"Resuming training from Epoch {self.epochs_ran}")

    def _save_snapshot(self, epoch: int) -> None:
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RAN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        distributed_print(
            f"Snapshot saved to ./{self.snapshot_file_path}\n",
            show_gpu=True,
            show_timestamp=True,
        )

    # ========== Reading to .csv files ==========
    def _read_columns_from_csv(self, file_name: str = "values.csv", directory="data"):
        file_path = os.path.join(directory, file_name)
        distributed_print(
            f"Reading values from ./{file_path}",
            show_gpu=True,
            debug=True,
        )
        df = pd.read_csv(file_path)
        return tuple(df[col] for col in df.columns)

    def _read_and_plot(self, csv_file_name: str, plot_file_name: str):
        batch_y_pred, targets = self._read_columns_from_csv()
        self._create_plot(batch_y_pred, targets)

    # ========== Writing to .csv files ==========
    def _write_tensors_to_csv(
        self,
        batch_y_pred,
        targets,
        file_name: str = "values.csv",
        directory="data",
    ):
        batch_y_pred, targets = self._tensors_to_numpy(batch_y_pred, targets)
        data = {
            "pred": batch_y_pred,
            "true": targets,
        }
        df = pd.DataFrame(data)

        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, file_name)
        distributed_print(f"Writing values to ./{file_path}", show_gpu=True, debug=True)
        df.to_csv(file_path, index=False)

    def _write_and_plot(
        self,
        csv_file_name: str,
        plot_file_name: str,
        y_true: torch.Tensor,
        y_pred: torch.Tensor = None,
        lower_bounds: torch.Tensor = None,
        upper_bounds: torch.Tensor = None,
        batch_index: int = 0,
        instance_index: int = 0,
    ):
        self._write_tensors_to_csv(
            csv_file_name,
            y_true,
            y_pred,
            lower_bounds,
            upper_bounds,
            batch_index,
            instance_index,
        )
        self._create_plot(
            y_true,
            y_pred,
            lower_bounds,
            upper_bounds,
            batch_index,
            instance_index,
            file_name=plot_file_name,
        )

    # ========== plotting ==========
    def _create_plot(
        self,
        batch_y_pred,
        targets,
        file_name="my_plot.png",
        directory="plots",
    ):
        # If batch_y_pred and targets are tensors, convert them into numpy arrays
        if torch.is_tensor(batch_y_pred) and torch.is_tensor(targets):
            batch_y_pred, targets = self._tensors_to_numpy(batch_y_pred, targets)

        # Create plot
        plt.figure()
        plt.plot(targets, label="True", linewidth=2)
        plt.plot(batch_y_pred, label="Pred", linewidth=2)
        plt.title("True vs Predicted Values")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

        # Save plot
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, file_name)
        distributed_print(f"Saving plot to ./{file_path}", show_gpu=True, debug=True)
        plt.savefig(file_path, bbox_inches="tight")

    def _tensors_to_numpy(self, batch_y_pred, targets):
        batch_y_pred = batch_y_pred.view(-1).cpu().numpy()
        targets = targets.view(-1).cpu().numpy()
        return batch_y_pred, targets

    def _get_settings(args, itr, seq_len=336):
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

    def _forward_pass(
        self,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        seq_trend,
        seq_seasonal,
        seq_resid,
    ):
        local_loss = None
        if self.args.model == "TEMPO" or "multi" in self.args.model:
            batch_y_pred, local_loss = self.model(
                batch_x,
                self.iteration,
                seq_trend,
                seq_seasonal,
                seq_resid,
            )
        elif "former" in self.args.model:
            dec_inp = (
                torch.zeros_like(batch_y[:, -self.args.pred_len :, :])
                .float()
                .to(self.local_rank)
            )
            dec_inp = (
                torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                .float()
                .to(self.local_rank)
            )
            batch_y_pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            batch_y_pred = self.model(batch_x, self.iteration)
        return batch_y_pred, local_loss

    def _compute_batch_loss(self, batch_y, batch_y_pred, local_loss):
        if self.args.loss_func == "prob" or self.args.loss_func == "negative_binomial":
            batch_y = batch_y[:, -self.args.pred_len :, :].squeeze()
            loss = self.criterion(batch_y, batch_y_pred)
        else:
            batch_y_pred = batch_y_pred[:, -self.args.pred_len :, :]
            batch_y = batch_y[:, -self.args.pred_len :, :]
            loss = self.criterion(batch_y_pred, batch_y)

        if self.args.model == "GPT4TS_multi" or self.args.model == "TEMPO_t5":
            if not self.args.no_stl_loss:
                loss += self.args.stl_weight * local_loss

        return loss

    def _unpack_tensor(self, data):
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

        # Move tensors to GPU
        batch_x = batch_x.float().to(self.local_rank)
        batch_y = batch_y.float().to(self.local_rank)
        batch_x_mark = batch_x_mark.float().to(self.local_rank)
        batch_y_mark = batch_y_mark.float().to(self.local_rank)
        seq_trend = seq_trend.float().to(self.local_rank)
        seq_seasonal = seq_seasonal.float().to(self.local_rank)
        seq_resid = seq_resid.float().to(self.local_rank)

        return (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            seq_trend,
            seq_seasonal,
            seq_resid,
        )

    def _run_batch(
        self,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        seq_trend,
        seq_seasonal,
        seq_resid,
    ):
        # Clear gradients (this line does nothing if torch.no_grad() is set)
        self.optimizer.zero_grad()

        # Compute forward pass
        batch_y_pred, local_loss = self._forward_pass(
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            seq_trend,
            seq_seasonal,
            seq_resid,
        )

        # Compute batch loss
        loss = self._compute_batch_loss(
            batch_y,
            batch_y_pred,
            local_loss,
        )

        batch_size = batch_x.size(0)
        return loss, batch_size

    def _tensors_to_numpy(
        y_true: torch.Tensor,
        y_pred: torch.Tensor = None,
        lower_bounds: torch.Tensor = None,
        upper_bounds: torch.Tensor = None,
        batch_index: int = 0,
        instance_index: int = 0,
    ):
        y_true = y_true[batch_index][instance_index].cpu().numpy().squeeze()
        if y_pred is not None:
            y_pred = y_pred[batch_index][instance_index].cpu().numpy().squeeze()
        if lower_bounds is not None:
            lower_bounds = lower_bounds[batch_index][instance_index].cpu().numpy()
        if upper_bounds is not None:
            upper_bounds = upper_bounds[batch_index][instance_index].cpu().numpy()
        return y_true, y_pred, lower_bounds, upper_bounds

    def _get_plot_name(self):
        prefix = "det" if self.args.loss_func == "mse" else "prob"
        return f"{prefix}_true_vs_pred"

    def _create_plot(
        self,
        y_true,
        y_pred=None,
        lower_bounds=None,
        upper_bounds=None,
        batch_index=0,
        instance_index=0,
        confidence_level: int = 95,
        file_name: str = None,
        directory: str = "plots",
    ):
        # If y_true is a tensor, then the rest of the iterables are most likely
        # tensors too, so convert them to numpy arrays
        if torch.is_tensor(y_true):
            y_true, y_pred, lower_bounds, upper_bounds = self._tensors_to_numpy(
                y_true,
                y_pred,
                lower_bounds,
                upper_bounds,
                instance_index,
                batch_index,
            )

        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create plot
        plt.figure()
        plt.plot(y_true, label="True", linewidth=2)

        if y_pred is not None:
            plt.plot(y_pred, label="Predicted", linewidth=2)

        if lower_bounds is not None and upper_bounds is not None:
            plt.fill_between(
                x=range(len(y_pred)),
                y1=lower_bounds,
                y2=upper_bounds,
                color="orange",
                alpha=0.2,
                label=f"Prediction Interval ({confidence_level}% confidence)",
            )

        plt.title("True vs Predicted Values")
        plt.xlabel("time")
        plt.ylabel("y")
        plt.legend()

        # If no file name is given, then generate name based on loss function
        if file_name is None:
            file_name = self._get_plot_name()
        plot_file_path = os.path.join(directory, f"{file_name}.png")

        # Save plot
        plt.savefig(plot_file_path, bbox_inches="tight")

    def _read_from_csv(self, file_name: str, directory: str = "results"):
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)
        return tuple(df[col] for col in df.columns)

    def _write_tensors_to_csv(
        self,
        file_name: str,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        lower_bounds: torch.Tensor = None,
        upper_bounds: torch.Tensor = None,
        batch_index: int = 0,
        instance_index: int = 0,
        directory="results",
    ):
        y_true, y_pred, lower_bounds, upper_bounds = self._tensors_to_numpy(
            y_true,
            y_pred,
            lower_bounds,
            upper_bounds,
            batch_index,
            instance_index,
        )

        data = {
            "pred": y_pred,
            "true": y_true,
        }

        if lower_bounds is not None and upper_bounds is not None:
            data["lower"] = lower_bounds
            data["upper"] = upper_bounds

        df = pd.DataFrame(data)
        file_path = os.path.join(directory, file_name)
        df.to_csv(file_path, index=False)

    def _test_probs(
        self,
        plot=True,
        plot_file_name=None,
        overwrite_csv=True,
        csv_file_name="prob_values.csv",
        batch_index=0,
        instance_index=0,
    ):
        if self.args.read_values and is_main_process():
            self._read_from_csv_and_plot(csv_file_name, plot_file_name)

        distributions = torch.tensor([], device=self.local_rank)
        y_pred = torch.tensor([], device=self.local_rank)
        y_true = torch.tensor([], device=self.local_rank)
        upper_bounds = torch.tensor([], device=self.local_rank)
        lower_bounds = torch.tensor([], device=self.local_rank)
        masks = torch.tensor([], device=self.local_rank)

        self.model.eval()
        with torch.no_grad():
            self.test_loader.set_epoch(self.iteration)

            if is_main_process():
                progress_bar = tqdm(total=len(self.test_loader))

            for data in self.test_loader:
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

                batch_x = batch_x.float().to(self.local_rank)
                batch_y = batch_y.float().to(self.local_rank)
                batch_x_mark = batch_x_mark.float().to(self.local_rank)
                batch_y_mark = batch_y_mark.float().to(self.local_rank)

                seq_trend = seq_trend.float().to(self.local_rank)
                seq_seasonal = seq_seasonal.float().to(self.local_rank)
                seq_resid = seq_resid.float().to(self.local_rank)

                num_channels = batch_x.shape[-1]

                # Compute channel-wise predictions
                for channel in range(num_channels):
                    probabilistic_forecasts = self.model.predict_prob(
                        batch_x[:, -self.args.seq_len :, channel : channel + 1],
                        num_samples=self.args.num_samples,
                        trend=seq_trend[:, -self.args.seq_len :, :],
                        seasonal=seq_seasonal[:, -self.args.seq_len :, :],
                        residual=seq_resid[:, -self.args.seq_len :, :],
                        pred_len=self.args.pred_len,
                    )

                    distributions = torch.cat((distributions, probabilistic_forecasts))

                    lower_bounds = torch.cat(
                        (
                            lower_bounds,
                            torch.quantile(probabilistic_forecasts, q=0.025, dim=0),
                        )
                    )

                    upper_bounds = torch.cat(
                        (
                            upper_bounds,
                            torch.quantile(probabilistic_forecasts, q=0.975, dim=0),
                        )
                    )

                    y_pred = torch.cat(
                        (
                            y_pred,
                            torch.mean(probabilistic_forecasts, dim=0),
                        )
                    )

                    y_true = torch.cat(
                        (y_true, batch_y[:, :, channel : channel + 1]),
                    )

                    masks = torch.cat(
                        (
                            masks,
                            batch_x_mark[
                                :, -self.args.pred_len :, channel : channel + 1
                            ],
                        )
                    )

                if is_main_process():
                    progress_bar.update(1)

        if is_main_process():
            progress_bar.close()

        # Aggregate tensors from all processes
        aggregated_distributions = aggregate_tensors(distributions)
        aggregated_y_true = aggregate_tensors(y_true)
        aggregated_y_pred = aggregate_tensors(y_pred)
        aggregated_lower_bounds = aggregate_tensors(lower_bounds)
        aggregated_upper_bounds = aggregate_tensors(upper_bounds)
        aggregated_masks = aggregate_tensors(masks)

        # Swap axes
        aggregated_y_true = torch.swapaxes(aggregated_y_true.squeeze(), -2, -3)
        unormzalized_gt_data = torch.swapaxes(aggregated_y_true.squeeze(), -1, -2)
        aggregated_masks = torch.swapaxes(aggregated_masks.squeeze(), -1, -2)
        target_mask = torch.swapaxes(aggregated_masks.squeeze(), -1, -2)

        crps_sum = calc_quantile_CRPS_sum(
            unormzalized_gt_data,
            aggregated_distributions,
            target_mask,
            mean_scaler=0,
            scaler=1,
        )

        crps = calc_quantile_CRPS(
            unormzalized_gt_data,
            aggregated_distributions,
            target_mask,
            mean_scaler=0,
            scaler=1,
        )

        if plot and overwrite_csv and is_main_process():
            self._write_to_csv_and_plot(
                csv_file_name,
                plot_file_name,
                aggregated_y_true,
                aggregated_y_pred,
                aggregated_lower_bounds,
                aggregated_upper_bounds,
                batch_index,
                instance_index,
            )

        return crps_sum, crps

    def _evaluate_val_set(self, epoch: int):
        # Total validation loss on a single process
        total_val_loss = 0.0

        # Number of samples seen on a single process
        num_samples = 0

        # Set model to evaluation mode
        self.model.eval()
        distributed_print(
            f"Evaluation mode: {not self.model.module.training}",
            debug=True,
        )

        with torch.no_grad():
            # Set epoch for distributed dataloader
            self.val_loader.sampler.set_epoch(epoch)

            # Only show progress bar on main process
            if is_main_process():
                progress_bar = tqdm(total=len(self.val_loader))

            for data in self.val_loader:
                # Run forward pass and compute batch loss
                loss, batch_size = self._run_batch(data)

                # Increment total validation loss and number of samples
                total_val_loss += loss.item()
                num_samples += batch_size

                if is_main_process():
                    progress_bar.update(1)  # Update progress bar

        if is_main_process():
            progress_bar.close()  # Close progress bar

        # Return average validation loss across all processses
        average_val_loss = aggregate_metrics(total_val_loss, num_samples)
        return average_val_loss
