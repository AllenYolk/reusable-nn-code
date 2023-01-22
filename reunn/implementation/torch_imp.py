from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils import tensorboard
from torch import optim
from tqdm import tqdm

from reunn.implementation import base_imp


class TorchPipelineImp(base_imp.BasePipelineImp):

    def __init__(
        self, net: nn.Module, log_dir: str,
        criterion: Optional[Callable] = None,
        optimizer: Optional[optim.Optimizer] = None,
        train_loader: Optional[data.DataLoader] = None, 
        test_loader: Optional[data.DataLoader] = None,
        validation_loader: Optional[data.DataLoader] = None,
    ):
        super().__init__()
        self.net = net
        self.log_dir = log_dir
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = None

    @staticmethod
    def acc_cnt(pred, labels) -> int:
        if pred.shape == labels.shape:
            labels = labels.argmax(dim=1)
        return (pred.argmax(dim=1) == labels).sum().item()

    def train_step(self, validation: bool = False, compute_acc: bool = False):
        train_loss, train_acc, train_sample_cnt = 0.0, None, 0
        if compute_acc:
            train_acc = 0

        self.net.train()
        for data, labels in tqdm(self.train_loader):
            pred = self.net(data)
            loss = self.criterion(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_sample_cnt += labels.shape[0]
            if compute_acc:
                train_acc += self.acc_cnt(pred, labels)

        train_loss /= train_sample_cnt
        if compute_acc:
            train_acc /= train_sample_cnt

        validation_loss, validation_acc = None, None
        if validation:
            validation_loss, validation_acc = self.validation_step(compute_acc)

        return train_loss, train_acc, validation_loss, validation_acc

    def _tv_step(self, data_loader: data.DataLoader, compute_acc: bool = False):
        accumulate_loss, accumulate_acc, accumulate_sample_cnt = 0.0, None, 0
        if compute_acc:
            accumulate_acc = 0

        self.net.eval()
        with torch.no_grad():
            for data, labels in data_loader:
                pred = self.net(data)
                loss = self.criterion(pred, labels)

                accumulate_loss += loss.item()
                accumulate_sample_cnt += labels.shape[0]
                if compute_acc:
                    accumulate_acc += self.acc_cnt(pred, labels)

        accumulate_loss /= accumulate_sample_cnt
        if compute_acc:
            accumulate_acc /= accumulate_sample_cnt

        return accumulate_loss, accumulate_acc

    def test_step(self, compute_acc: bool = False):
        return self._tv_step(self.test_loader, compute_acc)

    def validation_step(self, compute_acc: bool = False):
        return self._tv_step(self.validation_loader, compute_acc)

    def save_pipeline_state(
        self, dir: str, validation_loss: Optional[float] = None,
        validation_acc: Optional[float] = None,
        trained_epoch: Optional[int] = None,
    ):
        chk = {"state_dict": self.net.state_dict()}
        if validation_loss is not None:
            chk["validation_loss"] = validation_loss
        if validation_acc is not None:
            chk["validation_acc"] = validation_acc
        if trained_epoch is not None:
            chk["trained_epoch"] = trained_epoch
        torch.save(chk, dir)

    def load_pipeline_state(self, dir: str):
        chk = torch.load(dir)
        if "state_dict" in chk:
            self.net.load_state_dict(chk["state_dict"])
        return chk

    def add_runtime_records(self, main_tag, kv, idx):
        if self.writer is None:
            self.writer = tensorboard.SummaryWriter(self.log_dir)
        self.writer.add_scalars(main_tag, kv, idx)

    def clear_runtime_records(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None