import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torchsummary import summary
from mobilenet_v3 import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import shufflenet_v2_x0_5
from mydataset import mydataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    batch_size = 16
    epochs = 50
    train_dataset = mydataset(trans=True,
                              mode="train",)
    train_num = len(train_dataset)
    validate_dataset = mydataset(trans=True,
                              mode="val",)
    val_num = len(validate_dataset)


    # 定义数据读取方式
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, persistent_workers=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, persistent_workers=True)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = shufflenet_v2_x0_5()
    # net = mobilenet_v3_small(num_classes=10)
    # net = mobilenet_v3_large(num_classes=10)
    # net.load_state_dict(torch.load("./model/mobilenet_v3.pth"))
    # net.load_state_dict(torch.load("./model/mobilenet_v3_large.pth"))
    summary(net, (3, 224, 224), batch_size=1, device="cpu")
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    best_acc = 0.0
    save_path = './model/shufflenet.pth'
    # save_path = './model/mobilenet_v3_large.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            # labels = labels.float()
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate >= best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('Finished Training')


if __name__ == '__main__':
    train()
