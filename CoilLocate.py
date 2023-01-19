import torch
import numpy as np
import random
import cv2
import glob
import os
from builtins import str
from CoilNet import CoilNet
from ImageSet import *
from CoilImageProc import *


ROI = (1000, 0, 2048, 2048)


class CoilLocator(object):
    def __init__(self):
        network = CoilNet(img_size=512, output_size=2)
        network.add_conv_layer(channels=8, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=16, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=32, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=64, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=128, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=256, kernel_size=3, padding=1, pool_size=2)
        network.add_linear_layer_by_divider(divider=32)
        network.add_linear_layer(output_size=2)

        self.coil_net = network
        self.coil_net.init_model()
        self.proc_times = 0
        self.center = None
        self.filter_K = 0.8

        self.ROI = ROI     # left, top, width, height

    def load_model(self, model_file):
        self.coil_net.load_model(model_file)

    def resize_image_ROI(self, image):
        left = self.ROI[0]
        top = self.ROI[1]
        width = self.ROI[2]
        height = self.ROI[3]
        image = cv2.resize(image[top:top + height, left:left + width], (512, 512))
        return image

    def init_one_cycle(self):
        self.proc_times = 0

    def analyze_one_frame(self, frame):
        # 将待检测部分压缩至512×512
        frame = self.resize_image_ROI(frame)
        if len(frame.shape) > 2:
            frame = gray_image(frame)
        frame = norm_image(frame)

        # write images into a batch and predict the result
        batch_img = torch.zeros((1, 512, 512), dtype=torch.float32)
        batch_img[0] = torch.from_numpy(frame)
        params = self.coil_net.predict(batch_img)
        center = np.array((params[0][0], params[0][1])) * (self.ROI[2] / 512.0)
        center[0] += self.ROI[0]
        center[1] += self.ROI[1] + 100   # y方向定位有错误，暂时这样处理下
        self.center = center.astype(np.int32)

        cv2.circle(frame, (int(params[0][0]), int(params[0][1])), radius=5, color=255, thickness=2)
        cv2.imshow("", frame)

        # 对带卷中心位置进行滤波(第一次直接使用检测结果)
        #self.center = self.center * self.filter_K + center * (1.0 - self.filter_K) if self.proc_times else center

        self.proc_times += 1
        return self.center


def build_train_samples(dir_src, dir_dest):
    for file_name in glob.glob(dir_src + "\\*.jpg"):
        img = cv2.imread(file_name)

        left = ROI[0]
        top = ROI[1]
        width = ROI[2]
        height = ROI[3]
        img = img[top:top + height, left:left + width]
        img = cv2.resize(img, (512, 512))

        file_name = file_name.split('\\')[-1]
        params = extract_filename_params(file_name)
        cx = int(params[1]) - ROI[0]
        cy = int(params[2]) - ROI[1]
        long = int(params[3])
        short = int(params[4])

        params[1] = str(int(cx / 4))
        params[2] = str(int(cy / 4))
        params[3] = str(int(long / 4))
        params[4] = str(int(short / 4))

        file_name = ""
        for s in params:
            file_name += '-' + s
        file_name = dir_dest + "\\" + file_name[1: len(file_name)] + ".jpg"

        cv2.imwrite(file_name, img)


if __name__ == '__main__':
    # build_train_samples(dir_src="e:\\CoilLocate", dir_dest="e:\\CoilLocate\\resized")

    Net = CoilLocator().coil_net
    Net.learn_rate = 0.00001
    Net.l2_lambda = 0.0002
    Net.dropout_rate = 0.4
    Net.save_dir = ""
    Net.save_prefix = "带卷初始位置"
    Net.save_interval = 2000

    mode = input("Train(T) the model, or run(R) the model?")
    if mode == 't' or mode == 'T':
        train_set = ImageSet("E:\\CoilLocate\\resized",
                             output_size=LabelType.CenterOnly, img_size=512)
        train_set.random_noise = False
        train_set.batch_size = 100

        eval_set = ImageSet("E:\\CoilLocate\\resized",
                             output_size=LabelType.CenterOnly, img_size=512)
        eval_set.random_noise = False
        eval_set.batch_size = 100
        Net.train(train_set, eval_set, epoch_cnt=2000, mini_batch=64)
    else:
        Net.load_model("带卷初始位置2023-01-14-11-44.pt")
        eval_set = ImageSet("E:\\CoilLocate\\resized",   #"E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\InitPos",
                            output_size=LabelType.CenterOnly, img_size=512)
        eval_set.batch_size = 10
        eval_set.random_noise = False
        Net.test_net(eval_set)
        #Net.denoise_test(eval_set)