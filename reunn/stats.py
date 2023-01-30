from typing import Sequence


class NetStats:

    def __init__(self, net, input_shape: Sequence[int], backend: str = "torch"):
        if backend == "torch":
            from reunn.implementation import torch_imp 
            imp = torch_imp.TorchStatsImp(net, input_shape=input_shape)
        elif backend == "spikingjelly":
            from reunn.implementation import spikingjelly_imp
            imp = spikingjelly_imp.SpikingjellyStatsImp(net, input_shape)
        else:
            raise ValueError(f"{backend} backend not supported!")
        self.imp = imp

    def count_parameter(self):
        return self.imp.count_parameter()

    def count_mac(self):
        return self.imp.count_mac()

    def print_summary(self):
        self.imp.print_summary()
