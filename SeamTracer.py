import cv2
import torch
import glob
import os
import math
import numpy as np
import time
import random


import torch.nn as nn
import cv2
import numpy as np
from BaseCoilNet import BaseCoilNet

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        out=input.view(input.size(0), -1)
        return out


class Reshape(nn.Module): #将图像大小重定型
    def __init__(self, img_size):
        super(Reshape, self).__init__()
        self.img_size = img_size

    def forward(self, x):
        return x.view(-1, 1, self.img_size, self.img_size)      #(B x C x H x W)


class MyNet(nn.Module):
    def __init__(self, img_size, output_size, dropout_rate):
        super(MyNet, self).__init__()

        self.img_size = img_size
        self.output_size = output_size

        CONV1_CHANNELS = 8
        CONV2_CHANNELS = 16
        CONV3_CHANNELS = 32

        hidden1_input = int((img_size / (2**3))**2 * CONV3_CHANNELS)
        self.net = nn.Sequential(
            Reshape(img_size),
            # conv layer 1
            nn.Conv2d(in_channels=1, out_channels=CONV1_CHANNELS, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # conv layer 2
            nn.Conv2d(in_channels=CONV1_CHANNELS, out_channels=CONV2_CHANNELS, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # conv layer 3
            nn.Conv2d(in_channels=CONV2_CHANNELS, out_channels=CONV3_CHANNELS, kernel_size=5, stride=1, padding=2),
            nn.Dropout2d(dropout_rate),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # flatten
            Flatten(),
            # hidden layer 1
            nn.Linear(in_features=hidden1_input, out_features=int(hidden1_input / 2)),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            # output layer
            nn.Linear(in_features=int(hidden1_input / 2), out_features=output_size)
        )

    def forward(self, input):
        return self.net(input)

class SeamImageSet:
    def __init__(self, dir, img_size, y_range):
        self.img_size = img_size
        self.y_range = y_range

        self.images = list()
        os.chdir(dir)
        for file_name in glob.glob("*.jpg"):
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)  # read image from the file
            params = file_name.split('.')[0].split('-')
            label = int(params[1])
            self.images.append([img, label])

        print(("%d images have been loaded from " % len(self.images)) + dir)

    def random_sample(self, batch_size):
        sample_per_img = 8
        img_cnt = int(batch_size / sample_per_img)

        batch_img = torch.zeros(batch_size, self.img_size, self.img_size, dtype=torch.float32)
        batch_label = torch.zeros(batch_size, 1, dtype=torch.float32)
        raw_images = list()

        random.seed()
        for i in range(batch_size):
            idx = random.randint(0, len(self.images) - 1)
            offset = random.randint(0, 2 * self.img_size) - self.img_size

            cx = self.images[idx][1] - offset
            cy = random.randint(self.y_range[0], self.y_range[1])
            sub_img = self.images[idx][0][cy - int(self.img_size / 2): cy + int(self.img_size / 2),
                                            cx - int(self.img_size / 2): cx + int(self.img_size / 2)]

            label = -1
            if math.fabs(offset) < self.img_size / 2 - 2:
                label = float(offset) / self.img_size

            mean = sub_img.mean()
            std = sub_img.std()
            norm_img = (sub_img - mean) / std

            batch_img[i] = torch.from_numpy(norm_img)
            batch_label[i] = label
            raw_images.append(sub_img.copy())

        return batch_img, batch_label, raw_images

    def check_image_label(self, count):
        _, batch_label, raw_images = self.random_sample(count)
        for i, img in enumerate(raw_images):
            cx = int(self.img_size / 2)
            offset = int(batch_label[i] * self.img_size)
            cv2.line(img, (cx + offset, 20), (cx + offset, self.img_size - 20), 255, 2)
            cv2.putText(img, "%1.4f" % batch_label[i], (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
            cv2.imshow("", img)

            if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
                break

class SeamNet(BaseCoilNet):
    def __init__(self):
        super(SeamNet, self).__init__()

    def init_model(self):
        model = MyNet(64, 1, 0.4)
        super().init_model(model)

    def label2pixel(self, label):
        return int((label + 1.0) * self.model.img_size / 2)

    def norm_image(self, src):
        mean = src.mean()
        std = src.std()
        img = (src - mean) / std
        return img

    def img_window(self, center, img_size):
        left = int(int(center[0]) - img_size / 2)
        top = int(int(center[1]) - img_size / 2)
        right = int(int(center[0]) + img_size / 2)
        bottom = int(int(center[1]) + img_size / 2)

        return left, top, right, bottom

    def get_sub_image(self, img, center, img_size):
        left, top, right, bottom = self.img_window(center, img_size)

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

    def train(self, train_set, eval_set, epoch_cnt, mini_batch):
        loss_func = nn.MSELoss()
        super().train(loss_func, train_set, eval_set, epoch_cnt, mini_batch)

    def analyze_one_frame(self, frame, pos_y, x_begin, x_end, step):
        sub_imgs = list()
        for x in range(x_begin, x_end, step):
            img = self.get_sub_image(frame, (x, pos_y), self.model.img_size)
            sub_imgs.append(img)

        # write images into a batch and predict the result
        batch_img = torch.zeros(len(sub_imgs), self.model.img_size, self.model.img_size, dtype=torch.float32)
        for i, img in enumerate(sub_imgs):
            batch_img[i] = torch.from_numpy(self.norm_image(sub_imgs[i]))

        # average seam position
        pos_x = 0
        pos_cnt = 0
        outputs = self.predict(batch_img)
        for i in range(outputs.shape[0]):
            offset = outputs[i][0]
            if offset >= -0.5 and offset <= 0.5:
                x = x_begin + i * step + offset * self.model.img_size
                pos_x += x
                pos_cnt += 1
        if pos_cnt > 0:
            pos_x /= pos_cnt

        return int(pos_x)

    def test_net(self, eval_dir):
        os.chdir(eval_dir)
        for file_name in glob.glob("*.jpg"):
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)  # read image from the file
            pos_x = self.analyze_one_frame(img, 400, 350, 700, 8)
            cv2.line(img, (pos_x - 2, 0), (pos_x - 2, img.shape[0]), 255, 1)
            cv2.line(img, (pos_x + 2, 0), (pos_x + 2, img.shape[0]), 255, 1)
            cv2.imshow("", img)

            if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
                break

if __name__ == '__main__':
    Net = SeamNet()
    Net.init_model()

    mode = input('Train model(t) or test model(r)?')
    if mode == 't' or mode == 'T':
        train_set = SeamImageSet("D:/02 焊缝自动跟踪/训练集", 64, (350, 500))
        eval_set = train_set
        Net.save_dir = "D:/02 焊缝自动跟踪"
        Net.train(train_set, eval_set, 15000, 64)
    else:
        Net.load_model('D:/02 焊缝自动跟踪/2020-10-21-16-04.pt')
        Net.test_net("D:/02 焊缝自动跟踪/训练集")