import torch
import numpy as np
from torch import nn


def compress_linear(model, l=1):
    U, S, V = torch.svd(model.weight)
    # l = model.weight.shape[0] // l
    U1 = U[:, :l]
    S1 = S[:l]
    V1 = V[:, :l]
    V2 = torch.mm(torch.diag(S1), V1.T)
    new_model = nn.Sequential(nn.Linear(V2.shape[0], V2.shape[1], bias=False),
                              nn.Linear(U1.shape[0], U1.shape[1], bias=True))
    new_model[0].weight = torch.nn.parameter.Parameter(V2)
    new_model[1].bias = torch.nn.parameter.Parameter(torch.unsqueeze(model.bias, dim=0))
    new_model[1].weight = torch.nn.parameter.Parameter(U1)
    return new_model




if __name__ == '__main__':
    model = nn.Linear(4, 4, bias=True)
    new_model = compress_linear(model, l=1)

    input = torch.randn(4, 4)
    with torch.no_grad():
        output1 = model(input)
        output2 = new_model(input)
    print(f"output1 = {output1}")
    print(f"output2 = {output2}")