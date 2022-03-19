import albumentations as A
import cv2
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------数据增强-------------------------------------------------------
def get_normalize():
    return A.Normalize()

def get_train_transforms(p=1):
    return A.Compose(
        [A.Flip(),
         A.RandomBrightness(limit=(-0.2, 0.5), p=0.5),
         A.OneOf([
             A.IAAAdditiveGaussianNoise(),
             A.GaussNoise()], p=0.2),
         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.1),
         # A.OneOf([A.OpticalDistortion(p=0.3),
         #          A.GridDistortion(p=0.1),
         #          A.IAAPerspective(p=0.3)],
         #        p=0.2),
         A.OneOf([A.CLAHE(clip_limit=2),
                  A.IAASharpen(),
                  A.IAAEmboss()],
                  p=0.3),
         A.HueSaturationValue(p=0.3),
         A.JpegCompression(quality_lower=50, quality_upper=100, p=0.8),
         A.Normalize()
         ], p=p)


if __name__ == '__main__':
    trans = get_train_transforms()
    img = cv2.imread("../data_17/CV/imgs/img00001.jpg")
    img = trans(image=img)
    cv2.imshow("test", img['image'])
    cv2.waitKey(0)
