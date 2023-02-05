import abc
import os
from typing import Sequence


class BasePipelineImp(abc.ABC):

    def __init__(self, log_dir, hparam):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log_dir = log_dir
        self.hparam = hparam

    @staticmethod
    @abc.abstractmethod
    def acc_cnt(self, pred, labels):
        pass

    @abc.abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def validation_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save_pipeline_state(self, dir: str):
        pass

    @abc.abstractmethod
    def load_pipeline_state(self, dir: str):
        pass

    @abc.abstractmethod
    def add_runtime_records(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def add_hparam_records(self, *args, **kwargs):
        pass


class BaseStatsImp(abc.ABC):

    def __init__(self, net, input_shape: Sequence[int]):
        self.net = net
        self.input_shape = input_shape

    @abc.abstractmethod
    def count_parameter(self):
        pass

    @abc.abstractmethod
    def count_mac(self):
        pass

    @abc.abstractmethod
    def print_summary(self):
        pass
