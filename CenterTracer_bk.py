import cv2
import glob
import os
import math
import numpy as np
import time
from CoilNet import CoilNet
from ImageSet import LabelType
from CoilOffset import CoilOffset
from CoilTracer import CoilTracer
from ImageSet import ImageSet
from CoilImageProc import *
from ellipse import LsqEllipse


class CoilTracker(CoilTracer):
    def __init__(self, model_file):
        image_size = 512
        output_size = LabelType.InnerOnly

        network = CoilNet(img_size=image_size, output_size=output_size)
        network.add_conv_layer(channels=8, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=16, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=32, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=64, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=128, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=256, kernel_size=3, padding=1, pool_size=2)
        network.add_linear_layer_by_divider(divider=32)
        network.add_linear_layer(output_size=output_size)

        super().__init__(network, output_size)
        self.load_model(model_file)

        self.ellipse_with_noise = None

    def pre_process_image(self, image):
        edge = edge_image(image)

        if self.ellipse_with_noise is not None:
            ellipse = self.ellipse_with_noise
            mask = np.zeros(edge.shape, dtype=np.uint8)
            cv2.ellipse(mask, (int(edge.shape[0] / 2), int(edge.shape[1] / 2)), (ellipse[2], ellipse[3]), ellipse[4],
                        0, 360, 255, 32)
            mask = cv2.blur(mask, (32, 32), 0) / 255.0

            edge_noise = edge.copy()
            cv2.ellipse(edge_noise, (int(edge.shape[0] / 2), int(edge.shape[1] / 2)), (ellipse[2], ellipse[3]),
                        ellipse[4],
                        0, 360, 255, 2)
            edge = (edge * mask).astype(np.uint8)

            center, width, height, phi = fit_ellipse(edge, 50)
            img = edge.copy()
            cv2.ellipse(img, (int(center[0]), int(center[1])), (int(width), int(height)), int(phi), 0, 360, 255, 2)
            cv2.imshow("xxxx", img)

            return edge
        else:
            return edge


class CenterTracer(object):
    def __init__(self):
        self.WINDOW_SIZE = 512
        self.output_size = LabelType.InnerOnly

        self.tracker_with_noise = CoilTracker("带卷定位_有干扰_2022-08-05-16-05.pt")
        self.tracker_with_noise.window_name = "Edge with noise"
        self.tracker_with_noise.net_count = 4
        #self.tracker_with_noise.show_processing = True

        """
        self.tracker_no_noise = CoilTracker("带卷定位_无干扰_2022-08-06-22-02.pt")
        self.tracker_no_noise.window_name = "Edge without noise"
        self.tracker_no_noise.show_processing = True
        """

        # 通过帧差分检测偏移量时，需要确定检测窗口大小，和窗口相对初始中心点的偏移位置
        self.diff_size = 256           # 检测窗口尺寸，必须是2的指数
        self.diff_x_range = (-10, 10)
        self.diff_y_range = (-10, 10)
        self.diff_offset = (200, 0)
        self.lst_frame = None

        # 基于CNN和帧差分计算的方差估计值
        self.sigma_diff = 0.02
        self.sigma_cnn = 0.1

        self.record_trace = True
        self.repeat_trace = True
        self.kalman_trace = list()
        self.cnn_trace = list()

        # 对椭圆进行低通滤波
        self.lst_ellipses = None
        self.low_pass_k = 0.7

        self.show_fitting = True

    def init_one_cycle(self, init_center):
        frame_cnt = (self.diff_x_range[1] - self.diff_x_range[0]) * (self.diff_y_range[1] - self.diff_y_range[0])
        self.coil_diff = CoilOffset(self.diff_size ** 2, frame_cnt)

        self.init_center = init_center
        self.lst_frame = None
        self.sigma_kalman = self.sigma_cnn
        self.center_kalman = self.init_center
        self.lst_ellipses = None

        if self.repeat_trace:
            self.kalman_trace.append((-1, -1))
            self.cnn_trace.append((-1, -1))
        else:
            self.kalman_trace.clear()
            self.cnn_trace.clear()

        self.keytime = 10

    def offset_by_frame_diff(self, frame, lst_center, cur_center):
        if self.lst_frame is None:
            self.lst_frame = frame
            return 0, 0, 0

        # 从前一帧图像中截取差分检测窗口
        lst_center = (lst_center[0] + self.diff_offset[0], lst_center[1] + self.diff_offset[1])
        sub_img = get_sub_image(self.lst_frame, lst_center, self.diff_size)
        if sub_img is None:
            return 0, 0, 100
        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        lst_img = sub_img.reshape(self.diff_size * self.diff_size)

        # 从当前帧图像中截取一系列测试窗口，与前帧图像窗口做匹配，找出匹配度最高的窗口
        cur_center = (cur_center[0] + self.diff_offset[0], cur_center[1] + self.diff_offset[1])
        off_centers = trans_windows(cur_center, self.diff_x_range, self.diff_y_range, step=1)

        cur_imgs = np.zeros([len(off_centers), lst_img.shape[0]], np.uint8)
        for i, center in enumerate(off_centers):
            sub_img = get_sub_image(frame, center, self.diff_size)
            if sub_img is not None:
                sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
                cur_imgs[i] = sub_img.reshape(self.diff_size * self.diff_size).copy()

        # 计算每张图像差分后的误差平方和，从中选择误差最小的，最为最优匹配
        RMSEs = self.coil_diff.most_matching_frame(lst_img, cur_imgs)
        min_idx = np.argmin(RMSEs)
        off_x = off_centers[min_idx][0] - lst_center[0]
        off_y = off_centers[min_idx][1] - lst_center[1]

        # 保存当前帧用于下一次差分计算
        self.lst_frame = frame

        # 估算检测方差。如果图像未发生偏移，则认为检测方差为0
        sigma = 0.0 if off_x == 0 and off_y == 0 else self.sigma_diff

        return off_x, off_y, sigma

    def ellipse_by_fitting(self, frame, init_ellipse):
        init_center = (init_ellipse[0], init_ellipse[1])

        proc_image = get_sub_image(frame, center=init_center, img_size=self.WINDOW_SIZE)
        proc_image = edge_image(proc_image)

        init_ellipse[0] = init_ellipse[1] = int(self.WINDOW_SIZE / 2)
        proc_image = filter_around_ellipse(proc_image, init_ellipse)
        center, long, short, phi = fit_ellipse(proc_image, threshold=50)
        x, y = calc_original_pos(center, self.WINDOW_SIZE, init_center)
        ellipse = np.array((x, y, long, short, phi))

        if self.show_fitting:
            img = cv2.cvtColor(proc_image, cv2.COLOR_GRAY2BGR)
            cv2.ellipse(img, (int(center[0]), int(center[1])), (int(ellipse[2]), int(ellipse[3])), int(ellipse[4]),
                        0, 360, (0, 255, 0), 1)
            cv2.imshow("Ellipse fitting", img)

        # 对椭圆进行滤波，内圆中心点由卡尔曼滤波，不在这里处理
        if self.lst_ellipses is not None:
            for i in range(2, 5, 1):
                ellipse[i] = self.lst_ellipses[i] * self.low_pass_k + ellipse[i] * (1.0 - self.low_pass_k)
        self.lst_ellipses = ellipse

        return ellipse

    def analyze_one_frame(self, frame):
        # 基于CNN网络搜索带卷中心椭圆，由于噪音影响，该椭圆存在一定误差
        ellipse_cnn = self.tracker_with_noise.center_by_net(frame, self.WINDOW_SIZE, self.init_center)

        # 带卷运动到一定位置后，中心椭圆跟踪较为稳定，此时开始椭圆拟合进一步提高检测精度
        if ellipse_cnn[0] < 1900:
            ellipse_cnn = self.ellipse_by_fitting(frame, ellipse_cnn)

        sigma_cnn = self.sigma_cnn

        """
        if ellipse_cnn is None:
            ellipse_cnn = self.init_center
            sigma_cnn = 100
        """

        # locate coil offset by frame difference
        ox, oy, sigma_diff = self.offset_by_frame_diff(frame, lst_center=self.init_center, cur_center=ellipse_cnn)

        # final position by Kalman filter
        # calculate center by adding frame offset to the old center
        self.center_kalman = (self.center_kalman[0] + ox, self.center_kalman[1] + oy)
        self.sigma_kalman = math.sqrt(self.sigma_kalman**2 + sigma_diff**2)

        # combine center obtained by cnn
        kg = self.sigma_kalman**2 / (self.sigma_kalman**2 + sigma_cnn**2)
        self.center_kalman = (self.center_kalman[0] + kg * (ellipse_cnn[0] - self.center_kalman[0]),
                              self.center_kalman[1] + kg * (ellipse_cnn[1] - self.center_kalman[1]))
        self.sigma_kalman = math.sqrt((1.0 - kg) * self.sigma_kalman**2)

        self.init_center = (int(self.center_kalman[0] + 0.5), int(self.center_kalman[1] + 0.5))
        ellipse_cnn = ellipse_cnn.astype(np.int32)

        if self.record_trace == True:
            self.kalman_trace.append(self.init_center)
            self.cnn_trace.append(ellipse_cnn)

        return self.init_center, ellipse_cnn

    def show_trace(self, frame, center_kalman, center_cnn):
        # show coil center obtained by CNN and kalman filter
        cv2.circle(frame, (center_cnn[0], center_cnn[1]), 2, (255, 0, 0), 2)
        cv2.circle(frame, center_kalman, 2, (0, 255, 0), 2)

        # show ellipse
        if self.output_size > LabelType.CenterOnly and len(center_cnn) > 2:
            cv2.ellipse(frame, center_kalman, (center_cnn[2], center_cnn[3]), center_cnn[4], 0, 360, (0, 255, 0), 2)

        # show center moving trace
        lst_center = self.kalman_trace[0]
        for _, center in enumerate(self.kalman_trace):
            if lst_center[0] == -1:
                if center[0] != -1:
                    lst_center = center
            else:
                cv2.line(frame, lst_center, center, (0, 255, 0), 2)
                lst_center = center

        # 显示差分计算区域
        diff_center = (self.init_center[0] + self.diff_offset[0], self.init_center[1] + self.diff_offset[1])
        cv2.rectangle(frame,
                      (int(diff_center[0] - self.diff_size / 2), int(diff_center[1] - self.diff_size / 2)),
                      (int(diff_center[0] + self.diff_size / 2), int(diff_center[1] + self.diff_size / 2)),
                      (0, 0, 255), 2)


if __name__ == '__main__':
    Net = CenterTracer(output_size=LabelType.InnerOnly).coil_net
    Net.learn_rate = 0.00001
    Net.l2_lambda = 0.0002
    Net.dropout_rate = 0.4
    Net.save_dir = ""
    Net.save_prefix = "带卷定位"
    Net.save_interval = 2000

    mode = input("Train(T) the model, or run(R) the model?")
    if mode == 't' or mode == 'T':
        train_set = ImageSet("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\Train_no_noise",
                             output_size=LabelType.InnerOnly, img_size=512)
        train_set.random_noise = False
        train_set.batch_size = 1000

        eval_set = ImageSet("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\Eval_no_noise",
                             output_size=LabelType.InnerOnly, img_size=512)
        eval_set.random_noise = False
        eval_set.batch_size = 1000
        Net.train(train_set, eval_set, epoch_cnt=10000, mini_batch=64)
    else:
        Net.load_model("带卷定位2022-08-06-22-02.pt")
        eval_set = ImageSet("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\Eval_no_noise",
                            output_size=LabelType.InnerOnly, img_size=512)
        eval_set.random_noise = False
        Net.test_net(eval_set)
        #Net.denoise_test(eval_set)