import torch
from torch import nn
from torchvision import models
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
import random
import nni
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from nni.algorithms.compression.pytorch.pruning import L2FilterPruner
from nni.compression.pytorch import ModelSpeedup

epochs = 10
device = torch.device("cpu")
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor()
                     ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False,
                     transform=transforms.Compose([
                         transforms.ToTensor()
                     ])),
    batch_size=batch_size, shuffle=True)


class simple_model(nn.Module):
    def __init__(self):
        super(simple_model, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 5))
        )
        self.classifer = nn.Sequential(
            nn.Linear(64*5*5, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x


def train(model, mode="train", prune=False):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criteon = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if mode == "train":
                logits = model(data.to(device))
                loss = criteon(logits, target.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 1000 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                logits = model(data.to(device))
                test_loss += criteon(logits, target.to(device)).item()
                pred = logits.data.max(1)[1]
                target = target.cpu()
                pred = pred.cpu()
                correct += pred.eq(target.data).sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    if mode == "train":
        if prune:
            torch.save(model.state_dict(), "./model/prune_model.pth")
        else:
            torch.save(model.state_dict(), "./model/origin_model.pth")



if __name__ == '__main__':
    model = simple_model()
    train(model, mode="train")
    model.load_state_dict(torch.load("./model/origin_model.pth"))
    summary(model, (3, 64, 64), device="cpu")
    print("----------------???????????????-----------------")
    train(model, mode="val")
    config_list = [{'sparsity': 0.8, 'op_types': ['Conv2d']}]
    pruner = L2FilterPruner(model, config_list)
    model = pruner.compress()
    pruner.export_model(model_path="./model/prune.pth", mask_path="./model/mask.pth")
    pruner._unwrap_model()
    m_Speedup = ModelSpeedup(model, torch.randn([64, 3, 64, 64]), "./model/mask.pth", "cpu")
    m_Speedup.speedup_model()
    summary(model, (3, 64, 64), device="cpu")
    print("---------------??????????????????------------------")
    train(model, mode="val")
    train(model, mode="train", prune=True)
    print("---------------??????????????????------------------")
    train(model, mode="val")







