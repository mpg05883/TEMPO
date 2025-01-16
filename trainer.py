import os

import torch
import torch.nn.functional as F
from torch.distributed import all_gather, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        args,
        config,
        train_data: Dataset,
        train_loader: DataLoader,
        test_data: Dataset,
        test_loader: DataLoader,
        vali_data: Dataset,
        vali_loader: DataLoader,
        iteration: int,
        snapshot_path: str = "./snapshots",
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.args = args
        self.config = config
        self.train_data = train_data
        self.train_loader = train_loader
        self.test_data = test_data
        self.test_loader = test_loader
        self.vali_data = vali_data
        self.vali_loader = vali_loader
        self.iteration = iteration
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_loader))[0])
        # update print statements to use global rank
        print(
            f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}"
        )
        self.train_loader.sampler.set_epoch(epoch)
        for source, targets in self.train_loader:
            # move tensors to local rank
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _evaluate(self, epoch: int, val=True):
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            # get appropriate dataset
            dataset = self.vali_loader if val else self.test_loader

            # set epoch
            dataset.sampler.set_epoch(epoch)

            if self.global_rank == 0:
                pbar = tqdm(total=len(dataset))

            # run each batch in dataset
            for source, targets in dataset:
                # use local rank for .to() call
                source = source.to(self.local_rank)
                targets = targets.to(self.local_rank)

                # forward pass
                output = self.model(source)

                # calculate loss
                loss = F.cross_entropy(output, targets)

                # increment total_loss and num_samples
                total_loss += loss.item()
                num_samples += targets.size(0)

                if self.global_rank == 0:
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())

        if self.global_rank == 0:
            pbar.close()

        # compute this machine's average loss
        average_loss = total_loss / num_samples
        tensor = torch.tensor(
            average_loss, dtype=torch.float32, device=self.global_rank
        )
        print(
            f"[GPU{self.global_rank}]: Average loss = {tensor.item():.3e} = {total_loss:.3e}/{num_samples:.0f}"
        )

        # create list of tensors to aggregate average loss from all machines
        num_gpus = torch.cuda.device_count()
        tensor_list = [torch.zeros(1, device=self.local_rank) for _ in range(num_gpus)]

        # aggregate tensors from all machines
        all_gather(tensor_list, tensor)

        # get aggregated average loss
        stacked_tensor = torch.stack(tensor_list)
        aggregated_average_loss = torch.mean(stacked_tensor)

        if self.global_rank == 0:
            dataset_name = "validation" if val else "test"
            print(
                f"[GPU{self.global_rank}] | Average {dataset_name} loss: "
                f"{aggregated_average_loss:.3e}"
            )

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)

            # tell both GPUs to evaluate model
            if epoch % self.save_every == 0:
                self._evaluate(epoch, val=True)
                # only save snapshot from GPU 0
                if self.global_rank == 0:
                    self._save_snapshot(epoch)

            # tell GPU 1 to wait for GPU 0 to save a snapshot
            barrier()
            barrier()
            barrier()
