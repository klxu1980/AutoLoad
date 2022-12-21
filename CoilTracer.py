import torch
import numpy as np
import random
import cv2
from ImageSet import LabelType
from CoilImageProc import *


class CoilTracer(object):
    def __init__(self, network, output_size):
        self.output_size = output_size
        self.coil_net = network
        self.coil_net.init_model()

        # 使用CNN确定中心位置时，在初始位置附近，随机产生一系列窗口，对窗口中的图像检测中心，然后求平均
        self.net_x_range = (-20, 20)          # 随机窗口x方向范围
        self.net_y_range = (-20, 20)          # 随机窗口y方向范围
        self.net_count = 8                   # 随机窗口数量

        self.init_center = None

        self.trace = list()
        self.record_trace = True
        self.repeat_trace = True

        self.saving_video = False
        self.video_save = None

        self.show_processing = False
        self.window_name = "CoilTracer"

    def load_model(self, model_file):
        self.coil_net.load_model(model_file)

    def pre_process_image(self, image):
        return edge_image(image)

    def center_by_net(self, frame, img_size, init_center):
        # 预先提取图像边缘以提高精度
        # 为了减小计算耗时，仅对目标附近的区域进行边缘提取
        proc_image = get_sub_image(frame, center=init_center, img_size=img_size + 100)
        proc_image = self.pre_process_image(proc_image)

        # a set of images are clipped around the init center, each one is used to obtain one center location
        sub_center = (int((img_size + 100) / 2), int((img_size + 100) / 2))
        w_center = trans_windows_rand(sub_center, self.net_x_range, self.net_y_range, self.net_count)

        sub_imgs = list()
        for _, center in enumerate(w_center):
            img = get_sub_image(proc_image, center, img_size)
            if img is not None:
                sub_imgs.append((norm_image(img), center))

        rst_cnt = len(sub_imgs)
        if rst_cnt == 0:
            return None

        # write images into a batch and predict the result
        batch_img = torch.zeros(len(sub_imgs), int(img_size), int(img_size), dtype=torch.float32)
        for i, img in enumerate(sub_imgs):
            batch_img[i] = torch.from_numpy(img[0])
        params = self.coil_net.predict(batch_img)

        # average obtained results
        values = np.zeros(LabelType.InnerOuter, np.int32)
        for i, p in enumerate(params):
            x, y = calc_original_pos(p, img_size, sub_imgs[i][1])
            values[0] += x
            values[1] += y

            if self.output_size > LabelType.CenterOnly:
                values[2] += p[2]
                values[3] += p[3]
                values[4] += p[4]

            if self.output_size > LabelType.InnerOnly:
                values[5] += p[5] - img_size / 2 + sub_imgs[i][1][0]
                values[6] += p[6] - img_size / 2 + sub_imgs[i][1][1]
                values[7] += p[7]
                values[8] += p[8]
                values[9] += p[9]
        for i in range(10):
            values[i] = int(values[i] / rst_cnt + 0.5)

        if self.show_processing:
            proc_image = cv2.cvtColor(proc_image, cv2.COLOR_GRAY2BGR)
            cv2.ellipse(proc_image, (values[0], values[1]),
                        (values[2], values[3]), values[4], 0, 360, (0, 255, 0), 1)
            cv2.imshow(self.window_name, proc_image)

        # 计算目标在整个图像中的位置
        values[0] += init_center[0] - sub_center[0]
        values[1] += init_center[1] - sub_center[1]

        return values

    def init_one_cycle(self, init_center):
        self.init_center = init_center
        self.trace.clear()

    def show_trace(self, frame, center, color):
        # show coil center obtained by CNN and kalman filter
        cv2.circle(frame, center, 2, color, 2)

        # show center moving trace
        if len(self.trace):
            lst_c = self.trace[0]
            for _, c in enumerate(self.trace):
                cv2.line(frame, lst_c, c, color, 2)
                lst_c = c

        self.trace.append(center)
