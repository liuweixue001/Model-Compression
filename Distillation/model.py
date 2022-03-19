from torchvision import datasets
from tools.distiilled import DistillKL
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary
from resnet import resnet101
from mobilenet import mobilenet_v2
from torchvision import models



epochs = 100
distillkl = DistillKL(3)
device = torch.device("cuda")
batch_size = 8
image_size = (64, 64)
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Resize((224, 224)),
                         transforms.ToTensor()
                     ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('../data', train=False,
                     transform=transforms.Compose([
                         transforms.Resize((224, 224)),
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
            nn.Linear(64*5*5, 448),
            nn.Linear(448, 100),
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x


def train(model, distilled=False, model_name="simple_model", T=2):
    soft_loss = nn.KLDivLoss(reduction="batchmean")
    s_net = model().to(device)
    optimizer = optim.Adam(s_net.parameters(), lr=0.01)
    criteon = nn.CrossEntropyLoss()
    if distilled:
        t_net = resnet101().to(device)
        try:
            t_net.load_state_dict(torch.load("./model/resnet101.pth"))
            print("successful")
        except:
            print("failed")


    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # if batch_idx * len(data) >= 10000:
            #     break
            if distilled:
                with torch.no_grad():
                    t_logits = t_net(data.to(device))
            logits = s_net(data.to(device))
        if distilled:
            loss = 0.7 * criteon(logits, target.to(device)) + 0.3 * soft_loss(F.log_softmax(logits/T, dim=1),
                                                                              F.softmax(t_logits/T, dim=1))
            # loss = soft_loss(F.log_softmax(logits / T, dim=1), F.softmax(t_logits / T, dim=1))
        else:
            loss = criteon(logits, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


        test_loss = 0
        correct = 0
        for data, target in test_loader:
            logits = s_net(data.to(device))
            test_loss += criteon(logits, target.to(device)).item()
            pred = logits.data.max(1)[1]
            target = target.cpu()
            pred = pred.cpu()
            correct += pred.eq(target.data).sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        torch.save(s_net.state_dict(), f"./model/{model_name}.pth")


if __name__ == '__main__':
    train(simple_model, distilled=True, model_name="simple_model")