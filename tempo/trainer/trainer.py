import os
from argparse import Namespace

import torch
from omegaconf.dictconfig import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm
from utils.metrics import aggregate_metric
from utils.tools import adjust_learning_rate, print_rank_0, vali


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
        snapshot_path: str,
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
        self.snapshot_path = snapshot_path
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
        architecture: str,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        iteration,
        seq_trend,
        seq_seasonal,
        seq_resid,
    ):
        loss_local = None
        if architecture == "TEMPO" or "multi" in architecture:
            outputs, loss_local = self.model(
                batch_x,
                iteration,
                seq_trend,
                seq_seasonal,
                seq_resid,
            )
        elif "former" in architecture:
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
            outputs = self.model(batch_x, iteration)
        return outputs, loss_local

    def _compute_batch_loss(self, batch_y, outputs, loss_local):
        if self.args.loss_func == "prob" or self.args.loss_func == "negative_binomial":
            batch_y = batch_y[:, -self.args.pred_len :, :].squeeze()
            loss = self.criterion(batch_y, outputs)
        else:
            outputs = outputs[:, -self.args.pred_len :, :]
            batch_y = batch_y[:, -self.args.pred_len :, :]
            loss = self.criterion(outputs, batch_y)

        if self.args.model == "GPT4TS_multi" or self.args.model == "TEMPO_t5":
            if not self.args.no_stl_loss:
                loss += self.args.stl_weight * loss_local

        return loss.item()

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

        outputs, loss_local = self._forward_pass(
            self.args.model,
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            self.iteration,
            seq_trend,
            seq_seasonal,
            seq_resid,
        )

        loss = self._compute_batch_loss(
            batch_y,
            outputs,
            loss_local,
        )

        return loss

    def _test_probs(self, plot=True, values_file="prob_values.csv"):
        pass

    def test(self, plot=True, values_file="det_values.csv"):
        if self.args.loss_func != "mse":
            return self._test_probs()

        if self.args.read_values and self.global_rank == 0:
            pass

        self.model.eval()

        preds = torch.tensor([], dtype=torch.float32)
        trues = torch.tensor([], dtype=torch.float32)

        mse = MeanSquaredError()
        mae = MeanAbsoluteError()

        total_mae = 0.0
        total_mse = 0.0
        num_samples = 0

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
                    self.args.model,
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    self.iteration,
                    seq_trend,
                    seq_seasonal,
                    seq_resid,
                )

                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :]

                pred = outputs.detach()
                preds = torch.cat((preds, pred))

                true = batch_y.detach()
                trues = torch.cat((trues, true))

                batch_mae = mae(pred, true)
                batch_mse = mse(pred, true)

                batch_size = len(next(iter(self.test_loader))[0])
                total_mae += batch_mae * batch_size
                total_mse += batch_mse * batch_size
                num_samples += batch_size

                if self.global_rank == 0:
                    progress_bar.update(1)

        if self.global_rank == 0:
            progress_bar.close()

        # TODO: create plot

        average_mae = aggregate_metric(total_mae, num_samples, self.local_rank)
        average_mse = aggregate_metric(total_mse, num_samples, self.local_rank)

        return average_mae, average_mse

    def train(self):
        settings = self._get_settings(self.args, self.iteration)
        model_directory = os.path.join(self.args.checkpoints, settings)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        print_rank_0(f"Model will be saved to {model_directory}")

        for epoch in range(self.args.train_epochs):
            print_rank_0(
                f"\n========== Epoch {epoch + 1}/{self.args.train_epochs} =========="
            )

            total_train_loss = 0.0
            num_samples = 0

            self.train_loader.sampler.set_epoch(epoch)

            if self.global_rank == 0:
                progress_bar = tqdm(total=len(self.train_loader))

            for _, data in enumerate(self.train_loader):
                loss = self._run_batch(data)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                num_samples += len(next(iter(self.train_loader))[0])
                if self.global_rank == 0:
                    progress_bar.update(1)

            if self.global_rank == 0:
                progress_bar.close()

            average_train_loss = aggregate_metric(
                total_train_loss,
                num_samples,
                self.local_rank,
            )

            average_val_loss = vali(
                self.model,
                self.local_rank,
                self.global_rank,
                self.val_loader,
                self.criterion,
                self.args,
                self.iteration,
                epoch,
            )

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
                    f"\nEarlyStopping reached{self.early_stopping.counter}/{self.early_stopping.patience}"
                    "\nEnding training procedure early..."
                )
                return
