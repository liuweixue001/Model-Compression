from torch.utils.data import Dataset
import os
import cv2
import warnings
import numpy as np
import torch
import yaml
import albumentations as A
warnings.filterwarnings("ignore")


# -------------------------------------------------定义数据类---------------------------------------------------
class mydataset(Dataset):
    def __init__(self,
                 trans=True,
                 mode="train",
                 img_size=(224, 224)):
        # -------------------------------------------初始化数据&标签-----------------------------------------------------
        super().__init__()
        self.img_size = img_size
        self.trans = trans
        with open(mode + '.yaml', "r") as f1:
            content = f1.read()
            self.dir = yaml.load(content, Loader=yaml.FullLoader)
        with open(mode + '_label.yaml', "r") as f2:
            content = f2.read()
            self.label = yaml.load(content, Loader=yaml.FullLoader)

    # ----------------------------------------------------获取数据---------------------------------------------------
    def __getitem__(self, idx):
        imgsdir = self.dir[idx]
        img = cv2.imread(imgsdir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # --------------------------------------------数据增强----------------------------------------------------
        if self.trans:
            img = self.img_aug(img)
        # ---------------------------------------------生成数据-----------------------------------------------------
        img = cv2.resize(img, self.img_size)
        img = img.transpose(2, 0, 1).astype('float32')
        label = np.array(self.label[idx])
        return torch.tensor(img), torch.tensor(label.astype('int64'))

    # ----------------------------------------------------获取数据量-------------------------------------------------

    def __len__(self):
        return len(self.dir)

    # -----------------------------------------------------数据增强--------------------------------------------------
    def img_aug(self, img):
        return A.normalize(img=img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # -----------------------------------------------------图像显示--------------------------------------------------
    def show_normalize(self, idx):
        for i in idx:
            imgsdir = self.dir[i]
            img = cv2.imread(imgsdir)
            cv2.imshow('origin_img', img)
            cv2.moveWindow('origin_img', 0, 0)
            if self.trans:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.img_aug(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, self.img_size)
            cv2.imshow('normalized_img', img)
            cv2.moveWindow('normalized_img', 720, 0)
            cv2.waitKey(5000)


if __name__ == '__main__':
    a = mydataset(trans=True, mode="train")
    a.show_normalize([1, 2, 3, 4, 5, 6])
