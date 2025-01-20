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
from utils.metrics import aggregate_metrics, aggregate_tensors
from utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
    calc_quantile_CRPS,
    calc_quantile_CRPS_sum,
    print_rank_0,
)


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
    ):
        self.args = args
        self.config = config
        self.iteration = iteration
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.epochs_ran = 0
        self.snapshot_path = self.args.snapshot_path
        self._load_snapshot()

    def _load_snapshot(self):
        if not os.path.exists(self.snapshot_path):
            print_rank_0(
                "Cannot find previous snapshot. Starting training from epoch 0"
            )
            return

        map_location = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.snapshot_path, map_location=map_location)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_ran = snapshot["EPOCHS_RAN"]
        print_rank_0(f"Resuming training from Epoch {self.epochs_ran}")

    def _save_snapshot(self, epoch: int):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RAN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print_rank_0(f"Epoch {epoch} | Snapshot saved to {self.snapshot_path}")

    def load_checkpoint(self, checkpoint_path: str):
        print_rank_0(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        self.model.module.load_state_dict(checkpoint, strict=False)

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
            outputs, local_loss = self.model(
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
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, self.iteration)
        return outputs, local_loss

    def _compute_batch_loss(self, batch_y, outputs, local_loss):
        if self.args.loss_func == "prob" or self.args.loss_func == "negative_binomial":
            batch_y = batch_y[:, -self.args.pred_len :, :].squeeze()
            loss = self.criterion(batch_y, outputs)
        else:
            outputs = outputs[:, -self.args.pred_len :, :]
            batch_y = batch_y[:, -self.args.pred_len :, :]
            loss = self.criterion(outputs, batch_y)

        if self.args.model == "GPT4TS_multi" or self.args.model == "TEMPO_t5":
            if not self.args.no_stl_loss:
                loss += self.args.stl_weight * local_loss

        return loss

    def _run_batch(self, data):
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

        self.optimizer.zero_grad()

        outputs, local_loss = self._forward_pass(
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            seq_trend,
            seq_seasonal,
            seq_resid,
        )

        loss = self._compute_batch_loss(
            batch_y,
            outputs,
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

    def _read_from_csv_and_plot(self, csv_file_name: str, plot_file_name: str):
        y_true, y_pred, lower_bounds, upper_bounds = self._read_from_csv(csv_file_name)
        self._create_plot(y_true, y_pred, lower_bounds, upper_bounds, plot_file_name)

    def _write_to_csv_and_plot(
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
        self._create_plot(y_true, y_pred, lower_bounds, upper_bounds, plot_file_name)

    def _test_probs(
        self,
        plot=True,
        plot_file_name=None,
        overwrite_csv=True,
        csv_file_name="prob_values.csv",
        batch_index=0,
        instance_index=0,
    ):
        if self.args.read_values and self.global_rank == 0:
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

            if self.global_rank == 0:
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

                if self.global_rank == 0:
                    progress_bar.update(1)

        if self.global_rank == 0:
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

        if plot and overwrite_csv and self.global_rank == 0:
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

    def test(
        self,
        plot=True,
        plot_file_name=None,
        overwrite_csv=True,
        csv_file_name="det_values.csv",
        batch_index=0,
        instance_index=0,
    ):
        if self.args.loss_func != "mse":
            # Keep default csv file when name calling _test_probs()
            return self._test_probs(
                plot,
                plot_file_name,
                overwrite_csv,
                batch_index,
                instance_index,
            )

        if self.args.read_values and self.global_rank == 0:
            self._read_from_csv_and_plot(csv_file_name, plot_file_name)

        y_pred = torch.tensor([], device=self.local_rank)
        y_true = torch.tensor([], device=self.local_rank)

        total_mae = 0.0  # Total MAE for a single process
        total_mse = 0.0  # Total MSE for a single process
        num_samples = 0  # Number of samples seen on a single process

        self.model.eval()
        with torch.no_grad():
            self.test_loader.sampler.set_epoch(self.iteration)

            # Only display progress bar on rank 0 process
            if self.global_rank == 0:
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

                outputs, _ = self._forward_pass(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    seq_trend,
                    seq_seasonal,
                    seq_resid,
                )

                # Save last pred_len values
                outputs = outputs[:, -self.args.pred_len :, :].detach()
                batch_y = batch_y[:, -self.args.pred_len :, :].detach()
                y_pred = torch.cat((y_pred, outputs))
                y_true = torch.cat((y_true, batch_y))

                batch_mae = MeanAbsoluteError(outputs, batch_y)
                batch_mse = MeanSquaredError(outputs, batch_y)

                batch_size = batch_x.size(0)
                total_mae += batch_mae * batch_size
                total_mse += batch_mse * batch_size
                num_samples += batch_size

                if self.global_rank == 0:
                    progress_bar.update(1)

        if self.global_rank == 0:
            progress_bar.close()

        # Aggregate y_true and y_pred from all processes
        aggregated_y_true = aggregate_tensors(y_true)
        aggregated_y_pred = aggregate_tensors(y_pred)

        if plot and overwrite_csv and self.global_rank == 0:
            self._write_to_csv_and_plot(
                csv_file_name,
                plot_file_name,
                aggregated_y_true,
                aggregated_y_pred,
                batch_index,
                instance_index,
            )

        average_mae = aggregate_metrics(total_mae, num_samples)
        average_mse = aggregate_metrics(total_mse, num_samples)

        return average_mae, average_mse

    def _evaluate_validation_set(self, epoch: int):
        total_val_loss = 0.0
        num_samples = 0

        self.model.eval()
        with torch.no_grad():
            self.val_loader.set_epoch(epoch)

            if self.global_rank == 0:
                progress_bar = tqdm(total=len(self.val_loader))

            for _, data in enumerate(self.val_loader):
                loss, batch_size = self._run_batch(data)
                total_val_loss += loss.item()
                num_samples += batch_size

                if self.global_rank == 0:
                    progress_bar.update(1)

        if self.global_rank == 0:
            progress_bar.close()

        # Aggregate validation loss across all processes
        average_val_loss = aggregate_metrics(total_val_loss, num_samples)
        return average_val_loss

    def train(self):
        # Create directory to save model's checkpoint
        settings = self._get_settings(self.args, self.iteration)
        model_directory = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        print_rank_0(f"Model will be saved to {model_directory}")

        self.model.train()
        for epoch in range(self.epochs_ran, self.args.train_epochs):
            print_rank_0(
                f"\n========== Epoch {epoch + 1}/{self.args.train_epochs} =========="
            )

            # Total training loss on a single process
            total_train_loss = 0.0

            # Number of samples seen on a single process
            num_samples = 0

            self.train_loader.sampler.set_epoch(epoch)

            # Only display progress bar on global rank 0 process
            if self.global_rank == 0:
                progress_bar = tqdm(total=len(self.train_loader))

            for _, data in enumerate(self.train_loader):
                # Run forward pass and compute batch loss
                loss, batch_size = self._run_batch(data)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                num_samples += batch_size

                if self.global_rank == 0:
                    progress_bar.update(1)

            if self.global_rank == 0:
                progress_bar.close()

            # Aggregate average training losses across all processses
            average_train_loss = aggregate_metrics(total_train_loss, num_samples)

            # Compute average validation loss across all processses
            average_val_loss = self._evaluate_validation_set(epoch)

            print_rank_0(
                f"Train loss: {average_train_loss:.3f}"
                f" | Val loss: {average_val_loss:.3f}"
            )

            if self.args.cos:
                self.scheduler.step()
                print_rank_0(
                    "Learning rate: {:.3e}".format(self.optimizer.param_groups[0]["lr"])
                )
            else:
                adjust_learning_rate(self.optimizer, epoch + 1, self.args)

            self.early_stopping(
                average_val_loss,
                self.model,
                model_directory,
                self.global_rank,
            )
            if self.early_stopping.early_stop:
                print_rank_0(
                    f"\nEarlyStopping reached {self.early_stopping.counter}/{self.early_stopping.patience}"
                    "\nEnding training procedure early..."
                )
                return
            if self.early_stopping.early_stop:
                print_rank_0(
                    f"\nEarlyStopping reached {self.early_stopping.counter}/{self.early_stopping.patience}"
                    "\nEnding training procedure early..."
                )
                return
