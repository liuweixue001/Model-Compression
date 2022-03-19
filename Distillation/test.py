from torch import nn
from torchvision import models
from torchsummary import summary
# 加载预训练权重
model = models.resnet152(pretrained=True)
# 修改预测类别数
model.fc = nn.Linear(model.fc.in_features, 10)
summary(model, (3, 224, 224), device="cpu")