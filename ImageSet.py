import torch
import cv2
import numpy as np
import random
import glob
import os
import random
from enum import Enum
from CoilImageProc import *


# don't change to LabelType(Enum)
class LabelType():
    CenterOnly = 2
    InnerOnly  = 5
    InnerOuter = 10


class ImageSet:
    def __init__(self, output_size, img_size):
        self.raw_images = None
        self.norm_images = None
        self.img_size = img_size
        self.output_size = output_size
        self.batch_size = 1000
        self.reload_interval = 500
        self.read_times = 0
        self.random_noise = False
        self.edge_img = False
        self.file_list = list()

    def load(self, dir):
        self.file_list.clear()
        for file_name in glob.glob(dir + "\\*.jpg"):
            self.file_list.append(file_name)
        print(("There are %d images in " % len(self.file_list)) + dir)

    def load_random_batch(self):
        self.raw_images = list()
        self.norm_images = list()

        for i in range(self.batch_size):
            idx = random.randint(0, len(self.file_list) - 1)
            file_name = self.file_list[idx]
            if os.path.exists(file_name):
                img, norm_img, label, norm_label = self.read_labeled_image(file_name)
                if img is not None:
                    # add image to the list
                    self.raw_images.append([img, label])
                    self.norm_images.append([norm_img, norm_label])
        print("%d images have been loaded" % len(self.raw_images))

    def read_labeled_image(self, file_name):
        # load image from file and resize it
        src = cv_read(file_name, gray=True)
        img = cv2.resize(src, (self.img_size, self.img_size))
        if self.edge_img:
            img = edge_image(img)

        resize_k = src.shape[0] / self.img_size

        # extract label from file name
        # there are at most 11 contents in the file name:
        # the date time of the file, the center and shape of the inner ellipse, and those of the outer ellipse
        params = extract_filename_params(file_name=file_name)
        if len(params) <= self.output_size:
            return None

        label = np.zeros(self.output_size, dtype=np.int32)
        norm_label = np.zeros(self.output_size, dtype=np.float32)
        for i in range(self.output_size):
            label[i] = int(params[i + 1])
            if i % 5 < 4:
                label[i] = int(label[i] / resize_k + 0.5)   # the center and axis length
                norm_label[i] = self.pixel2label(label[i])
            else:
                norm_label[i] = self.angle2label(label[i])   # the angle of the ellipse

        # normalize image
        mean = img.mean()
        std = img.std()
        norm_img = ((img - mean) / std).astype(np.float32)

        return img, norm_img, label, norm_label

    def pixel2label(self, pixel):
        return pixel * 2.0 / self.img_size - 1.0

    @staticmethod
    def angle2label(angle):
        angle = angle % 360
        if angle >= 180:
            angle -= 180
        return angle / 90.0 - 1.0

    def label2pixel(self, label):
        return int((label + 1.0) * self.img_size)

    @staticmethod
    def label2angle(label):
        return int((label + 1.0) * 90.0)

    def random_sample(self, batch_size):
        if self.raw_images is None or self.read_times > self.reload_interval:
            self.load_random_batch()
            self.read_times = 0
        self.read_times += 1

        batch_img = torch.zeros(batch_size, self.img_size, self.img_size, dtype=torch.float32)
        batch_label = torch.zeros(batch_size, self.output_size, dtype=torch.float32)
        raw_images = list()

        random.seed()
        for i in range(batch_size):
            idx = random.randint(0, len(self.raw_images) - 1)
            img = self.raw_images[idx][0].copy()

            if self.random_noise:
                self.add_random_noise(img)
            norm_img = norm_image(img)

            batch_img[i] = torch.from_numpy(norm_img)
            batch_label[i] = torch.from_numpy(self.norm_images[idx][1])
            raw_images.append(img)

        return batch_img, batch_label, raw_images

    def random_cut_away(self):
        pass

    @staticmethod
    def add_random_line(image):
        x1 = random.randint(0, image.shape[1])
        y1 = random.randint(0, image.shape[0])
        x2 = random.randint(0, image.shape[1])
        y2 = random.randint(0, image.shape[0])
        color = random.randint(0, 255)
        width = random.randint(2, 10)
        cv2.line(image, (x1, y1), (x2, y2), color, width)

    def add_random_lines(self, image):
        cnt = random.randint(0, 3)
        for i in range(cnt):
            self.add_random_line(image)

    @staticmethod
    def add_random_arc(image):
        x = random.randint(0, image.shape[1])
        y = random.randint(0, image.shape[0])
        axis_l = random.randint(0, int(image.shape[1] / 3))
        axis_s = random.randint(0, int(image.shape[1] / 3))
        angle = random.randint(0, 360)
        start = random.randint(0, 360)
        end = random.randint(0, 360)
        color = random.randint(0, 255)
        width = random.randint(2, 5)
        cv2.ellipse(image, (x, y), (axis_l, axis_s), angle, start, end, color, width)

    def add_random_arcs(self, image):
        cnt = random.randint(0, 2)
        for i in range(cnt):
            self.add_random_arc(image)

    def add_random_noise(self, image):
        noise = np.zeros(image.shape, dtype=np.uint8)
        self.add_random_lines(noise)
        self.add_random_arcs(noise)
        noise = cv2.GaussianBlur(noise, (13, 13), 0)

        mask = noise > 5
        image[mask] = noise[mask]

    def check_image_label(self):
        for i, img_lbl in enumerate(self.raw_images):
            if len(img_lbl[0].shape) < 3:
                img = cv2.cvtColor(img_lbl[0].copy(), cv2.COLOR_GRAY2BGR)
            else:
                img = img_lbl[0].copy()

            cv2.circle(img, (img_lbl[1][0], img_lbl[1][1]), 5, (0, 255, 0), 1)
            if self.output_size > LabelType.CenterOnly:
                cv2.ellipse(img, (img_lbl[1][0], img_lbl[1][1]), (img_lbl[1][2], img_lbl[1][3]), img_lbl[1][4], 0, 360,
                            (0, 255, 0), 1)
            if self.output_size > LabelType.InnerOnly:
                cv2.ellipse(img, (img_lbl[1][5], img_lbl[1][6]), (img_lbl[1][7], img_lbl[1][8]), img_lbl[1][9], 0, 360,
                            (0, 255, 0), 1)
            cv2.imshow("", img)
            if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
                break

    def enum_images(self):
        for _, file in enumerate(self.file_list):
            img, norm_img, label, norm_label = self.read_labeled_image(file)
            cv2.circle(img, (label[0], label[1]), 5, (0, 255, 0), 1)
            cv2.imshow("", img)
            cv2.waitKey(0)

    def random_subset(self, sample_ration, remove=True):
        new_set = ImageSet(output_size=self.output_size, img_size=self.img_size)
        sample_list = random.sample(range(0, len(self.file_list)), int(len(self.file_list) * sample_ration) )
        for _, i in enumerate(sample_list):
            new_set.file_list.append(self.file_list[i])

        if remove:
            for _, i in enumerate(sample_list):
                self.file_list[i] = ""
            self.file_list = list(filter(lambda x: x != "", self.file_list))

        return new_set


if __name__ == '__main__':
    #img_set = ImageSet("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练集\\smallcoildata\\test", output_size=LabelType.InnerOnly, img_size=512)
    #img_set = ImageSet("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\Train", output_size=LabelType.InnerOnly, img_size=512)
    img_set = ImageSet("E:\\CoilLocate\\resized", output_size=LabelType.InnerOnly, img_size=512)
    #img_set = ImageSet("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\InitPos", output_size=LabelType.CenterOnly, img_size=512)
    #img_set.check_image_label()
    img_set.random_noise = False
    img_set.batch_size = 5
    while True:
        _, _, imgs = img_set.random_sample(1)
        img_set.check_image_label()
        cv2.imshow("", imgs[0])
        cv2.waitKey(0)