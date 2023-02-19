from typing import Optional, Callable, Sequence
import os

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils import tensorboard
from torch import optim
from tqdm import tqdm
import fvcore.nn

from reunn.implementation import base_imp


class TorchPipelineImp(base_imp.BasePipelineImp):

    def __init__(
        self, net: nn.Module, log_dir: str, hparam: dict,
        criterion: Optional[Callable] = None,
        optimizer: Optional[optim.Optimizer] = None,
        train_loader: Optional[data.DataLoader] = None, 
        test_loader: Optional[data.DataLoader] = None,
        validation_loader: Optional[data.DataLoader] = None,
    ):
        super().__init__(log_dir, hparam)
        self.net = net
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

    def data_label_process(self, data, labels, running_mode):
        return data, labels

    def pred_process(self, pred, running_mode):
        return pred

    def loss_process(self, loss, running_mode):
        return loss

    def after_batch_process(self, running_mode):
        pass

    def train_step(
        self, validation: bool = False, compute_acc: bool = False, 
        silent: bool = False,
    ):
        train_loss, train_acc, train_sample_cnt = 0.0, None, 0
        if compute_acc:
            train_acc = 0

        self.net.train()
        iterable = self.train_loader if silent else tqdm(self.train_loader)
        for data, labels in iterable:
            data, labels = self.data_label_process(data, labels, "train")
            pred = self.net(data)
            pred = self.pred_process(pred, "train")
            loss = self.criterion(pred, labels)
            loss = self.loss_process(loss, "train")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.after_batch_process("train")

            train_loss += loss.item() * labels.shape[0]
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

    def _tv_step(
        self, mode: str, data_loader: data.DataLoader, compute_acc: bool = False
    ):
        accumulate_loss, accumulate_acc, accumulate_sample_cnt = 0.0, None, 0
        if compute_acc:
            accumulate_acc = 0

        self.net.eval()
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = self.data_label_process(data, labels, mode)
                pred = self.net(data)
                pred = self.pred_process(pred, mode)
                loss = self.criterion(pred, labels)
                loss = self.loss_process(loss, mode)
                self.after_batch_process(mode)

                accumulate_loss += loss.item() * labels.shape[0]
                accumulate_sample_cnt += labels.shape[0]
                if compute_acc:
                    accumulate_acc += self.acc_cnt(pred, labels)

        accumulate_loss /= accumulate_sample_cnt
        if compute_acc:
            accumulate_acc /= accumulate_sample_cnt

        return accumulate_loss, accumulate_acc

    def test_step(self, compute_acc: bool = False):
        return self._tv_step("test", self.test_loader, compute_acc)

    def validation_step(self, compute_acc: bool = False):
        return self._tv_step("validation", self.validation_loader, compute_acc)

    def save_pipeline_state(
        self, file: str, validation_loss: Optional[float] = None,
        validation_acc: Optional[float] = None,
        train_loss: Optional[float] = None, train_acc: Optional[float] = None,
        min_loss_type: Optional[str] = None, max_acc_type: Optional[str] = None,
        trained_epoch: Optional[int] = None,
    ):
        chk = {"state_dict": self.net.state_dict(), "hparam": self.hparam}
        if validation_loss is not None:
            chk["validation_loss"] = validation_loss
        if validation_acc is not None:
            chk["validation_acc"] = validation_acc
        if train_loss is not None:
            chk["train_loss"] = train_loss
        if train_acc is not None:
            chk["train_acc"] = train_acc
        if trained_epoch is not None:
            chk["trained_epoch"] = trained_epoch
        if min_loss_type is not None:
            chk["min_loss_type"] = min_loss_type
        if max_acc_type is not None:
            chk["max_acc_type"] = max_acc_type

        dir = os.path.join(self.log_dir, file)
        torch.save(chk, dir)

    def load_pipeline_state(self, file: str):
        dir = os.path.join(self.log_dir, file)
        chk = torch.load(dir)
        if "state_dict" in chk:
            self.net.load_state_dict(chk["state_dict"])
        if "hparam" in chk:
            self.hparam = chk["hparam"]
        return chk

    def add_runtime_records(self, main_tag, kv, idx):
        if self.writer is None:
            self.writer = tensorboard.SummaryWriter(self.log_dir)
        self.writer.add_scalars(main_tag, kv, idx)

    def add_hparam_records(self, metrics: dict):
        if self.writer is None:
            self.writer = tensorboard.SummaryWriter(self.log_dir)
        self.writer.add_hparams(self.hparam, metrics)

    def _close_writer(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def __del__(self):
        self._close_writer()


class TorchStatsImp(base_imp.BaseStatsImp):

    def __init__(self, net, input_shape: Sequence[int]):
        super().__init__(net, input_shape)
        self._flop_count_analysis = None

    @property
    def flop_count_analysis(self):
        if self._flop_count_analysis is None:
            self._flop_count_analysis = fvcore.nn.FlopCountAnalysis(
                model=self.net, inputs=torch.randn(size=self.input_shape)
            )
        return self._flop_count_analysis

    def count_parameter(self):
        return fvcore.nn.parameter_count(self.net)[""]

    def count_mac(self):
        return self.flop_count_analysis.total()

    def print_summary(self):
        print(fvcore.nn.flop_count_table(self.flop_count_analysis))
