from typing import Optional, Callable, Sequence

import torch.nn as nn
from torch import optim
from torch.utils import data
from spikingjelly.activation_based import functional

from reunn.implementation import torch_imp


class SpikingjellyActivationBasedPipelineImp(torch_imp.TorchPipelineImp):

    def __init__(
        self, net: nn.Module, step_mode: str, log_dir: str, hparam: dict, 
        device: str = "cpu",
        criterion: Optional[Callable] = None,
        optimizer: Optional[optim.Optimizer] = None,
        train_loader: Optional[data.DataLoader] = None,
        test_loader: Optional[data.DataLoader] = None,
        validation_loader: Optional[data.DataLoader] = None,
    ):
        super().__init__(
            net, log_dir, hparam, device, criterion, optimizer, 
            train_loader, test_loader, validation_loader
        )
        self.step_mode = step_mode

    def after_batch_process(self, running_mode):
        return functional.reset_net(self.net)


class SpikingjellyStatsImp(torch_imp.TorchStatsImp):

    def __init__(self, net, input_shape: Sequence[int]):
        super().__init__(net, input_shape)
