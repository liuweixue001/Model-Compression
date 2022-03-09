from torch import nn
from decomposition_conv import tucker_decomposition_conv_layer
from torchvision import models
import torch
from torchsummary import summary
from decomposition_linear import compress_linear


def get_name(model):
    for name, module in model.named_children():
        pname = name
    return pname


def decomposition_conv2D(model):
    for name, module in model.named_children():
        pname = get_name(model)
        if name == pname:
            try:
                for i in range(len(list(module))):
                    if isinstance(module[i], nn.Conv2d):
                        try:
                            print(f"old_module = {module[i]}")
                            module[i] = tucker_decomposition_conv_layer(module[i])
                            print(f"new_module = {module[i]}")
                            print("this conv_layer successed")
                            return decomposition_conv2D(model)
                        except:
                            print("this conv_layer failed")
                            continue
                return model
            except:
                return model
        else:
            try:
                for i in range(len(list(module))):
                    if isinstance(module[i], nn.Conv2d):
                        try:
                            print(f"old_module = {module[i]}")
                            module[i] = tucker_decomposition_conv_layer(module[i])
                            print("this conv_layer successed")
                            print(f"new_module = {module[i]}")
                            return decomposition_conv2D(model)
                        except:
                            print("this conv_layer failed")
                            continue
            except:
                continue


def decomposition_linear(model, l):
    for name, module in model.named_children():
        pname = get_name(model)
        if name == pname:
            try:
                for i in range(len(list(module))):
                    if isinstance(module[i], nn.Linear):
                        try:
                            print(f"old_module = {module[i]}")
                            module[i] = compress_linear(module[i], l)
                            print(f"new_module = {module[i]}")
                            print("this linear_layer successed")
                            return decomposition_linear(model, l)
                        except:
                            print("this linear_layer failed")
                            continue
                return model
            except:
                return model
        else:
            try:
                for i in range(len(list(module))):
                    if isinstance(module[i], nn.Linear):
                        try:
                            print(f"old_module = {module[i]}")
                            module[i] = compress_linear(module[i], l)
                            print("this linear_layer successed")
                            print(f"new_module = {module[i]}")
                            return decomposition_linear(model, l)
                        except:
                            print("this linear_layer failed")
                            continue
            except:
                continue


def decomposition_model(model, conv2D=True, Linear=True, l=1):
    if conv2D:
        model = decomposition_conv2D(model)
    if Linear:
        model = decomposition_linear(model, l)
    return model


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
            nn.Linear(64*5*5, 20),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    model = simple_model()
    print("--------初始模型--------")
    summary(model, (3, 32, 32), device="cpu")
    try:
        model.load_state_dict(torch.load("./model/model.pth"))
        print("weights load successed")
    except:
        print("weights load failed")
    model = decomposition_model(model, l=5)
    print("--------压缩模型--------")
    summary(model, (3, 32, 32), device="cpu")
    torch.save(model.state_dict(), "model.pth")