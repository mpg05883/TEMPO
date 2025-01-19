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
from utils.metrics import aggregate_metric
from utils.tools import adjust_learning_rate, print_rank_0


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
        early_stopping,
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

    def _create_plot(
        self,
        y_true,
        y_pred=None,
        lower_bounds=None,
        upper_bounds=None,
        confidence_level=95,
        file_name="pred_vs_true.png",
    ):
        plots_directory = "plots"
        if not os.path.exists(plots_directory):
            os.makedirs(plots_directory)

        plt.figure()
        plt.title("Ground Truth vs Predicted Values")
        plt.plot(y_true, label="Ground Truth", linewidth=2)
        if y_pred is not None:
            plt.plot(y_pred, label="Predicted", linewidth=2)

            if lower_bounds is not None and upper_bounds is not None:
                x = range(len(y_pred))
                color = "orange"
                alpha = 0.2
                label = f"Prediction Interval ({confidence_level}% confidence)"
                plt.fill_between(
                    x=x,
                    y1=lower_bounds,
                    y2=upper_bounds,
                    color=color,
                    alpha=alpha,
                    label=label,
                )
        plt.xlabel("time")
        plt.ylabel("y")
        plt.legend()

        file_path = os.path.join(plots_directory, file_name)
        plt.savefig(file_path, bbox_inches="tight")

    def _test_probs(
        self,
        plot=True,
        results_directory="results",
        results_file="prob_values.csv",
        batch_index=0,
        instance_index=0,
    ):
        if self.args.read_values and self.global_rank == 0:
            results_path = os.path.join(results_directory, results_file)
            df = pd.read_csv(results_path)
            y_true = df["true"]
            y_pred = df["pred"]
            self._create_plot(y_true, y_pred)

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

            for _, data in enumerate(self.test_loader):
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

                torch.cuda.empty_cache()

                if self.global_rank == 0:
                    progress_bar.update(1)

        if self.global_rank == 0:
            progress_bar.close()

        # TODO: swap axes

        # TODO: compute local crps sum and crps

        # TODO: figure out how to compute aggregated metrics
        crps_sum, crps = 0, 0

        if plot and self.global_rank == 0:
            # TODO: figure out how to aggregate y_pred and y_true from all processes
            y_true = y_true[batch_index][instance_index].cpu().numpy().squeeze()
            y_pred = y_pred[batch_index][instance_index].cpu().numpy().squeeze()
            lower_bounds = lower_bounds[batch_index][instance_index].cpu().numpy()
            upper_bounds = upper_bounds[batch_index][instance_index].cpu().numpy()
            self._create_plot(y_pred, y_true, lower_bounds, upper_bounds)
            data = {
                "true": y_true,
                "pred": y_pred,
                "lower": lower_bounds,
                "upper": upper_bounds,
            }
            self._create_plot()
            df = pd.DataFrame(data)
            df.to_csv(results_path, index=False)

        return crps_sum, crps

    # * for now, I can just call test() from utils/tools.py
    def test(
        self,
        plot=True,
        results_directory="results",
        results_file="det_values.csv",
    ):
        if self.args.loss_func != "mse":
            return self._test_probs(plot, results_directory)

        if self.args.read_values and self.global_rank == 0:
            results_path = os.path.join(results_directory, results_file)
            df = pd.read_csv(results_path)
            y_true = df["true"]
            y_pred = df["pred"]
            self._create_plot(y_true, y_pred)

        y_pred = torch.tensor([], device=self.local_rank)
        y_true = torch.tensor([], device=self.local_rank)

        total_mae = 0.0
        total_mse = 0.0
        num_samples = 0

        self.model.eval()
        with torch.no_grad():
            self.test_loader.set_epoch(self.iteration)

            if self.global_rank == 0:
                progress_bar = tqdm(total=len(self.test_loader))

            for _, data in enumerate(self.test_loader):
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
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :]

                outputs = outputs.detach()
                y_pred = torch.cat((y_pred, outputs))

                batch_y = batch_y.detach()
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

        if plot and self.global_rank == 0:
            # TODO: figure out how to aggregate y_pred and y_true from all processes
            y_pred = y_pred.cpu().numpy().squeeze()
            y_true = y_true.cpu().numpy().squeeze()
            self._create_plot(y_pred, y_true)

            # Save pred and true values to .csv file for debugging
            data = {
                "pred": y_pred,
                "true": y_true,
            }
            df = pd.DataFrame(data)
            df.to_csv(results_path, index=False)

        average_mae = aggregate_metric(total_mae, num_samples)
        average_mse = aggregate_metric(total_mse, num_samples)

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
        average_val_loss = aggregate_metric(total_val_loss, num_samples)
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
            average_train_loss = aggregate_metric(total_train_loss, num_samples)

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
