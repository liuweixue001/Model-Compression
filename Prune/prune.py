import torch.nn.utils.prune as prune
from torch import nn


def model_prune(model, linear_percent=0.5, conv_percent=0):
    for layer in model.named_modules():
        for module in layer:
            # 全连接层剪枝
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=linear_percent)
                prune.remove(module, "weight")
            # 卷积层剪枝
            elif isinstance(module, nn.Conv2d):
                # pass
                prune.l1_unstructured(module, name="weight", amount=conv_percent)
                prune.remove(module, "weight")

def a(model):
    for name, layer in model.named_modules():
        prune.l1_unstructured(layer, name="weight", amount=0.9)
        prune.remove(layer, "weight")


if __name__ == '__main__':
    model = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)
    print(model.weight)
    a(model)
    print("====================")
    print(model.weight)