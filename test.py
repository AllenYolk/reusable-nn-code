import argparse

import torch
from torch import optim
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from reunn import pipeline


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*7*7, 10)
        )

    def forward(self, x):
        return self.f(x)


def train_test(data_dir, log_dir):
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
    net = Net()
    p = pipeline.SupervisedClassificationTaskPipeline(
        backend="torch", net=net, log_dir="../log_dir",
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(params=net.parameters(),lr=1e-4),
        train_loader=train_loader, validation_loader=validation_loader
    )
    p.train(epochs=50, validation=True, rec_best_checkpoint=True)


def load_test_test(data_dir, checkpoint_dir):
    test_loader = data.DataLoader(
        datasets.FashionMNIST(
            root=data_dir, train=False,
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=64, shuffle=True
    )
    net = Net()
    p = pipeline.SupervisedClassificationTaskPipeline(
        backend="torch", net=net, log_dir="../log_dir",
        criterion=nn.CrossEntropyLoss(),
        test_loader=test_loader
    )
    p.load_pipeline_state(dir=checkpoint_dir)
    p.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/")
    parser.add_argument("--log_dir", type=str, default="../log_dir")
    parser.add_argument("-m", "--mode", type=str, default="load")
    args = parser.parse_args()

    if args.mode == "train":
        train_test(args.data_dir, args.log_dir)
    elif args.mode == "load":
        load_test_test(args.data_dir, args.log_dir + "/best_checkpoint.pt")
