import cv2
import glob
import os
import numpy as np
import random
import datetime


def extract_filename(file_name, inner_ellipse):
    # extract file name for labels
    # there are 10 elements now in the file name:
    # the center, long and short axis, and angle of the inner ellipse
    # and those of the outer ellipse
    params = file_name.split('.')[0].split('-')
    if inner_ellipse and len(params) >= 6:
        label = np.array([int(params[1]), int(params[2]), int(params[3]), int(params[4]), int(params[5])])
    elif len(params) >= 11:
        label = np.array([int(params[6]), int(params[7]), int(params[8]), int(params[9]), int(params[10])])
    else:
        return None
    return label


def generate_image_by_trans(src_img, label, img_size, random_range, count, dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # generate train images with random offset
    random.seed()
    for i in range(count):
        x_offset = random.randint(-random_range, random_range)
        y_offset = random.randint(-random_range, random_range)
        x_center = label[0] + x_offset
        y_center = label[1] + y_offset

        if x_center - img_size[0] / 2 < 0 or x_center + img_size[0] / 2 > src_img.shape[1] or \
                y_center - img_size[1] / 2 < 0 or y_center + img_size[1] / 2 > src_img.shape[0]:
            continue

        # avoid offset out of src image bound
        '''
        if x_center - img_size[0] / 2 < 0:
            x_center = img_size[0] / 2
            x_offset = img_size[0] / 2 - label[0]
        elif x_center + img_size[0] / 2 > src_img.shape[1]:
            x_center = src_img.shape[1] - img_size[0] / 2 - 1
            x_offset = src_img.shape[1] - label[0] - img_size[0] / 2 - 1

        if y_center - img_size[1] / 2 < 0:
            y_center = img_size[1] / 2
            y_offset = img_size[1] / 2 - label[1]
        elif y_center + img_size[1] / 2 > src_img.shape[0]:
            y_center = src_img.shape[0] - img_size[1] / 2 - 1
            y_offset = src_img.shape[0] - label[1] - img_size[1] / 2 - 1
        '''

        # subtract image for training
        sub_img = src_img[int(y_center - img_size[1] / 2) : int(y_center + img_size[1] / 2),
                            int(x_center - img_size[0] / 2) : int(x_center + img_size[0] / 2)]

        now = datetime.datetime.now()
        datetime_str = "/%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        label_str = "-%04d-%04d-%04d-%04d-%03d.jpg" % (int(img_size[0] / 2 - x_offset), int(img_size[1] / 2 - y_offset),
                                                    label[2], label[3], label[4])
        cv2.imwrite(dst_dir + datetime_str + label_str, sub_img)


def generate_train_samples(src_dir, dst_dir, inner_ellipse):
    img_size = (512, 512) if inner_ellipse else (1400, 1400)

    os.chdir(src_dir + "/")
    for file_name in glob.glob("*.jpg"):
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        label = extract_filename(file_name, inner_ellipse)
        generate_image_by_trans(img, label, img_size, random_range=50, count=20, dst_dir=dst_dir)


def check_image_label(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    lbl_inner = extract_filename(file_name, inner_ellipse=True)
    if lbl_inner is not None:
        cv2.ellipse(img, (lbl_inner[0], lbl_inner[1]), (lbl_inner[2], lbl_inner[3]), lbl_inner[4], 0, 360, 0,
                    2)
        cv2.circle(img, (lbl_inner[0], lbl_inner[1]), 5, 0, 3)

    lbl_outer = extract_filename(file_name, inner_ellipse=False)
    if lbl_outer is not None:
        cv2.ellipse(img, (lbl_outer[0], lbl_outer[1]), (lbl_outer[2], lbl_outer[3]), lbl_outer[4], 0, 360, 0, 2)
        cv2.circle(img, (lbl_outer[0], lbl_outer[1]), 5, 0, 3)

    if img.shape[1] > 1024:
        k = 1024.0 / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1] * k), int(img.shape[0] * k)))
    cv2.imshow("labeled image", img)
    

def test_image_labels(dir):
    os.chdir(dir + "/")
    for file_name in glob.glob("*.jpg"):
        check_image_label(file_name)
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    #generate_train_samples("D:/01 自动上卷/标注图片", "D:/TrainOuter", inner_ellipse=True)
    test_image_labels("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\Train_raw")
'''
def main():
    print("Generate labeled training images(Y) or check labeled images(N)?")
    generate_train_samples("D:/01 自动上卷/标注图片", "D:/TrainOuter", inner_ellipse=True)
    test_image_labels("D:/TrainOuter")

if __name__ == '__main__':
    main()
'''

test_image_labels("D:/TrainOuter")