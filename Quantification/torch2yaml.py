import os
import yaml


class2label = {"bird": 0, "boat": 1, "cake": 2, "jellyfish": 3, "king_crab": 4}

def gen_imgdir(dir="train"):
    txt = []
    for classdir in os.listdir("./mini_imagenet/" + dir):
        for imgdir in os.listdir("./mini_imagenet/" + dir + "/" + classdir):
            txt.append("./mini_imagenet/" + dir + "/" + classdir + "/" + imgdir)
    with open(dir + '.yaml', "w", encoding="utf-8") as f1:
        yaml.dump(txt, f1)


def gen_label(dir="train"):
    txt = []
    with open(dir + ".yaml", "r") as f2:
        con = f2.read()
        C = yaml.load(con, Loader=yaml.FullLoader)
        for i in C:
            label = i.split("/")[3]
            txt.append(class2label[label])
    with open(dir + '_label.yaml', "w", encoding="utf-8") as f2:
        yaml.dump(txt, f2)



if __name__ == '__main__':
    # for i in {"train", "test", "val"}:
    #     dir = i
    #     gen_imgdir(dir)
    #     gen_label(dir)
    # with open('val.yaml', "r") as f:
    #     content = f.read()
    #     C = yaml.load(content, Loader=yaml.FullLoader)
    #     print(C[10])
    # txt = []
    # for classdir in os.listdir("./images"):
    #     txt.append("./images" "/" + classdir)
    # with open("./images/predict" + '.yaml', "w", encoding="utf-8") as f1:
    #     yaml.dump(txt, f1)
    # txt = []
    # with open("./images/predict" + '.yaml', "r") as f2:
    #     con = f2.read()
    #     C = yaml.load(con, Loader=yaml.FullLoader)
    #     for i in C:
    #         label = i.split("/")[3]
    #         txt.append(class2label[label])
    # with open("./images/predict" + '_label.yaml', "w", encoding="utf-8") as f2:
    #     yaml.dump(txt, f2)
    pass
