import cv2
import numpy as np
import os
import glob
edge_detector = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")


def edge_image(image):
    image = np.float32(image)
    image = image * (1.0 / 255.0)
    edge = edge_detector.detectEdges(image)

    max_value = np.max(edge)
    edge = edge * 255 / max_value
    edge = edge.astype(np.uint8)

    return edge


def remove_ellipse_noise(image, file_name):
    ellipse = extract_filename(file_name, inner_ellipse=True)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.ellipse(mask, (ellipse[0], ellipse[1]), (ellipse[2], ellipse[3]), ellipse[4], 0, 360, 255, 32)
    mask = cv2.blur(mask, (32, 32), 0) / 255.0
    image = (image * mask).astype(np.uint8)
    return image


def convert_images(src_dir, dst_dir, remove_noise=False):
    os.chdir(src_dir)
    for file_name in glob.glob("*.jpg"):
        img = cv2.imread(file_name)
        edge = edge_image(img)
        if remove_noise:
            edge = remove_ellipse_noise(edge, file_name)
        cv2.imwrite(dst_dir + "\\" + file_name, edge)


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


def img_window(center, img_size):
    left = int(int(center[0]) - img_size / 2)
    top = int(int(center[1]) - img_size / 2)
    right = int(int(center[0]) + img_size / 2)
    bottom = int(int(center[1]) + img_size / 2)

    return left, top, right, bottom


def get_sub_image(img, center, img_size):
    left, top, right, bottom = img_window(center, img_size)

    if left < 0:
        right += -left
        left = 0
    elif right > img.shape[1]:
        left -= right - img.shape[1]
        right = img.shape[1]

    if top < 0:
        bottom += -top
        top = 0
    elif bottom > img.shape[0]:
        top -= bottom - img.shape[0]
        bottom = img.shape[0]

    sub_img = img[top: bottom, left: right]
    return sub_img


def check_image_label(file_name):
    img = cv2.imread(file_name)
    label = extract_filename(file_name, inner_ellipse=True)
    edge = get_sub_image(img, (label[0], label[1]), 512)

    img = np.zeros((edge.shape[1], edge.shape[0] * 2, 3), dtype=np.uint8)
    img[:, 0: edge.shape[1]] = edge
    img[:, edge.shape[1]: edge.shape[1] * 2] = edge

    cv2.ellipse(img, (label[0], label[1]), (label[2], label[3]), label[4], 0, 360, (0, 255, 0), 1)
    cv2.imshow("", img)


def check_all_image_label(dir):
    os.chdir(dir)
    for file_name in glob.glob("*.jpg"):
        check_image_label(file_name)
        cv2.waitKey(0)


convert_images(src_dir="E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\Eval_raw", dst_dir="G:\\edge")
check_all_image_label("G:\\edge")