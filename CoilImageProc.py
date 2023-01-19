import cv2
import numpy as np
import random
import math
from ellipse import LsqEllipse     # pip install lsq-ellipse
from FilePath import PYTHON_PATH


# model.yml.gz文件就在项目文件夹下，在python环境下使用相对路径就可以了。当VC调用时，需要使用绝对路径
edge_detector = cv2.ximgproc.createStructuredEdgeDetection(PYTHON_PATH + "model.yml.gz")


class VideoPlayer(object):
    def __init__(self, file_name, ROI=None):
        self.file_name = file_name
        self.video = cv2.VideoCapture(file_name)
        self.ROI = ROI

        self.frame_cnt = int(self.video.get(7))
        self.cur_frame = 0

    def get_frame(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
        _, frame = self.video.read()
        if self.ROI is not None:
            frame = frame[self.ROI[1]: self.ROI[3], self.ROI[0]: self.ROI[2], :]
        return frame

    def forward(self):
        self.cur_frame = min(self.frame_cnt - 1, self.cur_frame + 1)

    def backward(self):
        self.cur_frame = max(0, self.cur_frame - 1)

    def is_end(self):
        return self.cur_frame >= self.frame_cnt - 1


def cv_read(file_name, gray=False):
    img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), -1)
    if gray and len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def cv_write(file_name, image):
    cv2.imencode('.jpg', image)[1].tofile(file_name)


def gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def color_image(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def edge_image_old(image):
    # For some unknown reason a grayed image will cause error in detectEdges(),
    # so a grayed image need be transformed to a colored image.
    if len(image.shape) == 2:
        image = color_image(image)
    image = np.float32(image) * (1.0 / 255.0)
    edge = edge_detector.detectEdges(image)
    max_value = np.max(edge)
    edge_img = (edge * 255 / max_value).astype(np.uint8)
    return edge_img


def edge_image(image):
    raw_img = image.copy()
    edge1 = edge_image_old(raw_img)

    # image = cv2.equalizeHist(image)
    # edge2 = edge_image_old(image)
    #
    # img = np.hstack((edge1, edge2))
    # cv2.imshow("", img)

    return edge1


def norm_image(image):
    mean = image.mean().astype(np.float32)
    std = image.std()
    norm_img = (image - mean) / std
    return norm_img


def sobel_image(image):
    x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 1)

    # 转换数据 并 合成
    Scale_absX = cv2.convertScaleAbs(x)  # 格式转换函数
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)  # 图像混合
    return result


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


def trans_windows(init_center, x_range, y_range, step):
    w_center = list()
    for x in range(int((x_range[1] - x_range[0]) / step)):
        for y in range(int((y_range[1] - y_range[0]) / step)):
            w_center.append([init_center[0] + int(x * step + x_range[0]), init_center[1] + int(y * step + y_range[0])])
    return w_center


def trans_windows_rand(init_center, x_range, y_range, count):
    w_center = list()
    for i in range(count):
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        w_center.append([init_center[0] + x, init_center[1] + y])
    return w_center


def filter_around_ellipse(image, ellipse, range=(0, 360), width=32, blur_width=32):
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.ellipse(mask, (ellipse[0], ellipse[1]), (ellipse[2], ellipse[3]), ellipse[4],
                range[0] - ellipse[4], range[1] - ellipse[4], 255, width)
    mask = cv2.blur(mask, (blur_width, blur_width), 0) / 255.0
    image = (image * mask).astype(np.uint8)
    return image


def fit_ellipse(image, threshold):
    xy = np.where(image > threshold)
    xy = np.array(list(zip(xy[1], xy[0])))
    try:
        reg = LsqEllipse().fit(xy)
    except:
        return None

    center, long, short, phi = reg.as_parameters()
    phi *= 180.0 / math.pi

    # 调整长短轴
    if long < short:
        long, short = short, long
        phi += 90.0

    # 角度限制在0-180之间
    phi = int(phi * 100.0) % 36000
    if phi > 18000:
        phi -= 18000
    phi *= 0.01

    ellipse = np.array((center[0], center[1], long, short, phi))
    return ellipse


def calc_original_pos(pos, img_size, init_center):
    x = pos[0] - img_size / 2 + init_center[0]
    y = pos[1] - img_size / 2 + init_center[1]
    return x, y


def extract_filename_params(file_name):
    params = file_name.split('.')[0].split('-')
    return params


def image_histogram(image):
    hist, _ = np.histogram(image.flatten(), 256)
    dist = hist / hist.max()
    return hist, dist


def draw_histogram(hist):
    img = np.zeros((100, 256), dtype=np.uint8)
    if hist is not None:
        for i, h in enumerate(hist):
            if not np.isnan(h):
                v = int(h * 100)
                img[100 - v: 100, i] = 255
    return img


if __name__ == '__main__':
    """
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(mask, (256, 256), (150, 100), 30, 150, 400, 255, 3)
    cv2.imshow("", mask)
    cv2.waitKey(0)
    """
    img = cv2.imread("e:\\xxx.jpg")
    cv2.imshow("xx", img)
    #img = gray_image(img)
    img = edge_image(img)
    cv2.imshow("", img)
    cv2.waitKey(0)