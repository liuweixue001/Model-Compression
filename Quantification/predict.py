import torch
from mydataset import mydataset
from torchvision.models import shufflenet_v2_x0_5
from tqdm import tqdm
import cv2


batch_size = 1
epochs = 50
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
class2label = {"bird": 0, "boat": 1, "cake": 2, "jellyfish": 3, "king_crab": 4}


def predict():
    model = shufflenet_v2_x0_5(num_classes=5).to(device)
    model_weight_path = "./model/shufflenet.pth"
    try:
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print("successed")
    except:
        print("failed to load model")
    print("using {} device.".format(device))
    test_dataset = mydataset(trans=True,
                                mode="./images/predict")
    # test_dataset = mydataset(trans=True,
    #                             mode="test")
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=1)
    test_num = len(test_dataset)
    model.eval()
    test_bar = tqdm(test_loader)
    acc = 0.0
    with torch.no_grad():
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_images_y =  torch.squeeze(test_images).permute(1, 2, 0).numpy()
            outputs = model(test_images.to(device))
            outputs = torch.squeeze(outputs)
            outputs = torch.softmax(outputs, dim=0)
            predict = torch.max(outputs, dim=0)[1]
            predict_y = predict.numpy()
            test_labels_y = test_labels.numpy()
            classname = list(class2label.keys())[list(class2label.values()).index(predict_y)]
            real_classname = list(class2label.keys())[list(class2label.values()).index(test_labels_y)]
            print(f"真实类别为: {real_classname}, 预测类别为: {classname}, 预测概率为: {outputs.numpy()[predict_y]:.3}, "
                  f"第一个点像数值为: {test_images_y[0, 0, :]}")
            # cv2.imshow("test", test_images_y)
            # cv2.waitKey(0)
            acc += torch.eq(predict, test_labels.to(device)).sum().item()
        test_accurate = acc / test_num
    print(f"test_accurate = {test_accurate:.3}")


if __name__ == '__main__':
    predict()
