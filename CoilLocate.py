import torch
import numpy as np
import random
import cv2
from CoilNet import CoilNet
from ImageSet import *
from CoilImageProc import *


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
        self.filter_K = 0.99

    def load_model(self, model_file):
        self.coil_net.load_model(model_file)

    def init_one_cycle(self):
        self.proc_times = 0

    def center_by_net(self, frame):
        resize_x = frame.shape[1] / 512
        resize_y = frame.shape[0] / 512

        if len(frame.shape) > 2:
            frame = gray_image(frame)
        frame = norm_image(cv2.resize(frame, (512, 512)))

        # write images into a batch and predict the result
        batch_img = torch.zeros((1, 512, 512), dtype=torch.float32)
        batch_img[0] = torch.from_numpy(frame)
        params = self.coil_net.predict(batch_img)
        center = np.array((params[0][0] * resize_x, params[0][1] * resize_y))

        if not self.proc_times:
            self.center = center
        else:
            self.center = self.center * self.filter_K + center * (1.0 - self.filter_K)

        self.proc_times += 1
        return self.center.astype(np.int32)


if __name__ == '__main__':
    Net = CoilLocator().coil_net
    Net.learn_rate = 0.00001
    Net.l2_lambda = 0.0002
    Net.dropout_rate = 0.4
    Net.save_dir = ""
    Net.save_prefix = "带卷初始位置"
    Net.save_interval = 2000

    mode = input("Train(T) the model, or run(R) the model?")
    if mode == 't' or mode == 'T':
        train_set = ImageSet("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\InitPos",
                             output_size=LabelType.CenterOnly, img_size=512)
        train_set.random_noise = False
        train_set.batch_size = 100

        eval_set = ImageSet("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\InitPos",
                             output_size=LabelType.CenterOnly, img_size=512)
        eval_set.random_noise = False
        eval_set.batch_size = 100
        Net.train(train_set, eval_set, epoch_cnt=10000, mini_batch=64)
    else:
        Net.load_model("带卷初始位置2022-08-18-22-55.pt")
        eval_set = ImageSet("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\InitPos",
                            output_size=LabelType.CenterOnly, img_size=512)
        eval_set.random_noise = False
        Net.test_net(eval_set)
        #Net.denoise_test(eval_set)