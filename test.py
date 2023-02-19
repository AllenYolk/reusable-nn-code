import argparse

from torch import optim
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import functional

import reunn


def train_test(data_dir, log_dir, epochs, silent):
    train_loader = data.DataLoader(
        datasets.FashionMNIST(
            root=data_dir, train=True, 
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=64, shuffle=True
    )
    validation_loader = data.DataLoader(
        datasets.FashionMNIST(
            root=data_dir, train=False, 
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=64, shuffle=True
    )
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    p = reunn.SupervisedClassificationTaskPipeline(
        backend="torch", net=net, log_dir=log_dir, hparam={"epochs": epochs},
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(params=net.parameters(),lr=1e-4),
        train_loader=train_loader, validation_loader=validation_loader
    )
    p.train(
        epochs=epochs, validation=True, rec_best_checkpoint=True, 
        rec_latest_checkpoint=True, rec_runtime_msg=True, silent=silent
    )


def load_test_test(data_dir, log_dir):
    test_loader = data.DataLoader(
        datasets.FashionMNIST(
            root=data_dir, train=False,
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=64, shuffle=True
    )
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    p = reunn.SupervisedClassificationTaskPipeline(
        backend="torch", net=net, log_dir=log_dir, hparam=None,
        criterion=nn.CrossEntropyLoss(),
        test_loader=test_loader
    )
    p.load_pipeline_state(file="best_checkpoint.pt")
    p.test()


def stats_test():
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    s = reunn.NetStats(net=net, input_shape=[1, 1, 28, 28])
    print(f"#parameters: {s.count_parameter()}")
    print(f"#MACs: {s.count_mac()}")
    s.print_summary()


def spikingjelly_test(data_dir, log_dir, epochs, T, silent):
    class multi_step_data_extend(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T
        def forward(self, x):
            l = len(x.shape)
            return x.repeat(T, *[1 for i in range(l)])

    class multi_step_pred_sum(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x.sum(dim=0)

    net = nn.Sequential(
        multi_step_data_extend(T),
        layer.Flatten(),
        layer.Linear(784, 512),
        neuron.IFNode(),
        layer.Linear(512, 128),
        neuron.IFNode(),
        layer.Linear(128, 10),
        multi_step_pred_sum()
    )
    functional.set_step_mode(net, "m")
    s = reunn.NetStats(
        net=net, input_shape=[1, 1, 28, 28], backend="spikingjelly"
    )
    s.print_summary()

    train_loader = data.DataLoader(
        datasets.MNIST(
            root=data_dir, train=True, 
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=64, shuffle=True
    )
    validation_loader = data.DataLoader(
        datasets.MNIST(
            root=data_dir, train=False, 
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=128, shuffle=True
    )
    p = reunn.SupervisedClassificationTaskPipeline(
        backend="spikingjelly", net=net, step_mode="m", log_dir=log_dir,
        hparam={"epochs": epochs, "T": T},
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(params=net.parameters(),lr=1e-4),
        train_loader=train_loader, validation_loader=validation_loader
    )
    p.train(
        epochs=epochs, validation=True, rec_best_checkpoint=False, 
        rec_latest_checkpoint=False, rec_runtime_msg=False, silent=silent
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/")
    parser.add_argument("--log_dir", type=str, default="../log_dir")
    parser.add_argument("-m", "--mode", type=str, default="spikingjelly")
    parser.add_argument("-s", "--silent", action="store_true")
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-T", type=int, default=4)
    args = parser.parse_args()

    if args.mode == "train":
        train_test(args.data_dir, args.log_dir, args.epochs, args.silent)
    elif args.mode == "load":
        load_test_test(args.data_dir, args.log_dir)
    elif args.mode == "stats":
        stats_test()
    elif args.mode == "spikingjelly":
        spikingjelly_test(
            args.data_dir, args.log_dir, args.epochs, args.T, args.silent,
        )
