import torch
from torch import optim
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from reunn import pipeline


DATA_DIR = "~/CodeRepo/datasets/CIFAR-10"

if __name__ == "__main__":
    train_loader = data.DataLoader(
        datasets.CIFAR10(
            root=DATA_DIR, train=True, 
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=64, shuffle=True
    )
    validate_loader = data.DataLoader(
        datasets.CIFAR10(
            root=DATA_DIR, train=False, 
            download=True, transform=transforms.ToTensor(),
        ),
        batch_size=64, shuffle=True
    )
    net = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(3*16*16, 10)
    )
    p = pipeline.SupervisedTaskPipeline(
        backend="torch", net=net,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(params=net.parameters(),lr=1e-4),
        train_loader=train_loader, validate_loader=validate_loader
    )
    p.train(epochs=50, validation=True, compute_acc=True)