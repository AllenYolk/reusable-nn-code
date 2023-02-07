from typing import Optional, Callable, Sequence

import torch.nn as nn
from torch import optim
from torch.utils import data
from spikingjelly.activation_based import functional

from reunn.implementation import torch_imp


class SpikingjellyActivationBasedPipelineImp(torch_imp.TorchPipelineImp):

    def __init__(
        self, net: nn.Module, T: int, log_dir: str, hparam: dict,
        criterion: Optional[Callable] = None,
        optimizer: Optional[optim.Optimizer] = None,
        train_loader: Optional[data.DataLoader] = None,
        test_loader: Optional[data.DataLoader] = None,
        validation_loader: Optional[data.DataLoader] = None,
    ):
        super().__init__(
            net, log_dir, hparam, criterion, optimizer, 
            train_loader, test_loader, validation_loader
        )
        self.T = T

    def data_label_process(self, data, labels, mode):
        functional.reset_net(self.net)
        n_dim_data = len(data.shape)
        data = data.repeat(self.T, *[1 for _ in range(n_dim_data)])
        return data, labels

    def pred_process(self, pred, mode):
        return pred.sum(dim = 0)


class SpikingjellyStatsImp(torch_imp.TorchStatsImp):

    def __init__(self, net, input_shape: Sequence[int]):
        super().__init__(net, input_shape)
