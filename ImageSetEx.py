import torch
import cv2
import numpy as np
import random
import glob
import os
from XMLImage import XmlImage


class ImageSet:
    def __init__(self, dir, img_size, grey_scaled=True, pre_proc=False):
        self.img_size = img_size
        self.labeled_images = []
        self.regen_interval = 500
        self.regen_count = 20
        self.regen_range = 32
        self.sample_times = 1
        self.images = None
        self.centers = None

        file_filter = dir + "\\*.xml" if len(dir) else "*.xml"
        for file_name in glob.glob(file_filter):
            labeled_image = XmlImage(file_name, grey_scaled, pre_proc)
            if labeled_image.image is not None:
                self.labeled_images.append(labeled_image)
        print("There are %d images in %s" % (len(self.labeled_images), dir))

        #self.regenerate_samples()
        #print("Generate random samples. %d samples are generated." % self.sample_count())

    def read_labeled_image(self, idx):
        img = self.images[idx]
        center = self.centers[idx]
        norm_center = (self.pixel2label(center[0]), self.pixel2label(center[1]))

        # normalize image
        mean = img.mean()
        std = img.std()
        norm_img = (img - mean) / std

        return img, norm_img, np.array(center), np.array(norm_center)

    def sample_count(self):
        return len(self.images)

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
    def label2angle(self, label):
        return int((label + 1.0) * 90.0)

    @staticmethod
    def alpha_beta_adjust(image, alpha, beta):
        img = image * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(dtype=np.uint8)

    def random_alpha_beta(self, image):
        alpha = random.randint(-100, 100) * 0.005 + 1
        beta = 0
        return self.alpha_beta_adjust(image, alpha, beta)

    def add_salt_noise(self, image):
        pass

    def regenerate_samples(self):
        self.images = []
        self.centers = []
        for _, img in enumerate(self.labeled_images):
            images, centers = img.gen_subimages_around_centers((self.img_size, self.img_size), self.regen_range,
                                                               self.regen_count)
            self.images += images
            self.centers += centers

    def random_sample(self, batch_size):
        if self.sample_times % self.regen_interval == 0:
            self.regenerate_samples()
            print("Regenerate random samples. %d samples are generated." % self.sample_count())
        self.sample_times += 1

        batch_img = torch.zeros(batch_size, self.img_size, self.img_size, dtype=torch.float32)
        batch_label = torch.zeros(batch_size, 2, dtype=torch.float32)
        raw_images = list()

        random.seed()
        img_cnt = self.sample_count()
        for i in range(batch_size):
            idx = random.randint(0, img_cnt - 1)
            raw, norm, raw_label, norm_label = self.read_labeled_image(idx)

            batch_img[i] = torch.from_numpy(norm)
            batch_label[i] = torch.from_numpy(norm_label)
            raw_images.append(raw)

        return batch_img, batch_label, raw_images

    def get_sample(self, idx):
        raw, norm, raw_label, norm_label = self.read_labeled_image(idx)

        batch_img = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        batch_label = torch.zeros(1, 2, dtype=torch.float32)

        batch_img[0] = torch.from_numpy(norm)
        batch_label[0] = torch.from_numpy(norm_label)

        return batch_img, batch_label, raw

    @staticmethod
    def __draw_image_center_only(img, label):
        cv2.circle(img, (label[0], label[1]), 5, 0, 3)

    def check_image_label(self):
        self.regenerate_samples()
        for idx in range(self.sample_count()):
            raw, norm, raw_label, norm_label = self.read_labeled_image(idx)

            img = raw.copy()
            self.__draw_image_center_only(img, raw_label)

            cv2.imshow("image", img)
            if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
                break


if __name__ == '__main__':
    img_set = ImageSet("E:\\01 我的设计\\01 一飞院项目\\012 加油机仿真\\01 程序输出\\01 机器人控制程序\\HoseImages", img_size=256)
    img_set.check_image_label()