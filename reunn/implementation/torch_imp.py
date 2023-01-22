from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
from tqdm import tqdm

from reunn.implementation import base_imp


class TorchPipelineImp(base_imp.BasePipelineImp):

    def __init__(
        self, net: nn.Module,
        criterion: Optional[Callable] = None,
        optimizer: Optional[optim.Optimizer] = None,
        train_loader: Optional[data.DataLoader] = None, 
        test_loader: Optional[data.DataLoader] = None,
        validate_loader: Optional[data.DataLoader] = None,
    ):
        super().__init__()
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.validate_loader = validate_loader
        self.optimizer = optimizer
        self.criterion = criterion

    @staticmethod
    def acc_cnt(pred, labels):
        if pred.shape == labels.shape:
            labels = labels.argmax(dim=1)
        return (pred.argmax(dim=1) == labels).sum().item()

    def train_step(self, validation: bool = False, compute_acc: bool = False):
        train_loss, train_acc, train_sample_cnt = 0.0, 0, 0

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
        print("train loss", train_loss)
        if compute_acc:
            train_acc /= train_sample_cnt
            print("train acc", train_acc)

        if validation:
            self.validate_step(compute_acc)

    def _tv_step(self, data_loader: data.DataLoader, compute_acc: bool = False):
        accumulate_loss, accumulate_acc, accumulate_sample_cnt = 0.0, 0, 0

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
        print(accumulate_loss)
        if compute_acc:
            accumulate_acc /= accumulate_sample_cnt
            print(accumulate_acc)

    def test_step(self, compute_acc: bool = False):
        self._tv_step(self.test_loader, compute_acc)

    def validate_step(self, compute_acc: bool = False):
        self._tv_step(self.validate_loader, compute_acc)
