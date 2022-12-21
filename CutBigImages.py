import cv2
import glob
from LabeledImage import LabeledImage


if __name__ == '__main__':
    for file_name in glob.glob("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷对准\\带卷对准训练\\*.xml"):
        labeled_image = LabeledImage(file_name, grey_scaled=False, pre_proc=False)
        if labeled_image.image is None:
            continue

        center = labeled_image.centers[0]
        img = labeled_image.image[center[1] - 256: center[1] + 256, center[0] - 256: center[0] + 256]
        labeled_image.image = img
        labeled_image.centers[0] = (256, 256)
        labeled_image.save()