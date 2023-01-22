import argparse

import torch
from torch import optim
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from reunn import pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../datasets/")
    args = parser.parse_args()

    train_loader = data.DataLoader(
        datasets.FashionMNIST(
            root=args.data_dir, train=True, 
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=64, shuffle=True
    )
    validation_loader = data.DataLoader(
        datasets.FashionMNIST(
            root=args.data_dir, train=False, 
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=64, shuffle=True
    )
    net = nn.Sequential(
        nn.Conv2d(1, 4, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 16, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16*7*7, 10)
    )
    p = pipeline.SupervisedClassificationTaskPipeline(
        backend="torch", net=net,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(params=net.parameters(),lr=1e-4),
        train_loader=train_loader, validation_loader=validation_loader
    )
    p.train(epochs=50, validation=True)