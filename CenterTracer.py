import cv2
import numpy as np
from filterpy.kalman import KalmanFilter       # pip install filterpy
from filterpy.common import Q_discrete_white_noise
import csv
from enum import Enum
from CoilNet import CoilNet
from ImageSet import LabelType
from ImageSet import ImageSet
#from CoilOffset import CoilOffset
from CoilTracer import CoilTracer
from CoilImageProc import *


class WarningType(Enum):
    ok = 0
    ellipse_fitting_failed = 1
    ellipse_large_error = 2


class CenterTracer(CoilTracer):
    def __init__(self):
        self.WINDOW_SIZE = 512
        self.output_size = LabelType.InnerOnly
        network = self.init_network(self.WINDOW_SIZE, self.output_size)
        super().__init__(network, self.output_size)

        # 通过帧差分检测偏移量时，需要确定检测窗口大小，和窗口相对初始中心点的偏移位置
        self.diff_size = 256           # 检测窗口尺寸，必须是2的指数
        #self.diff_x_range = (-10, 10)
        #self.diff_y_range = (-10, 10)
        self.diff_offset = (180, -180)
        self.lst_frame = None

        # 初始化卡尔曼滤波器
        self.init_kalman()

        # 对椭圆进行低通滤波
        self.noise_filter_begin = 1800     # 开始滤波范围，当带卷中心x坐标小于该值时，开始滤波
        self.exp_ellipse = None            # 当前位置预期的椭圆参数字典
        self.lst_ellipses = None           # 前一周期的椭圆
        self.low_pass_k = 0.7              # 椭圆参数滤波系数

        self.ellipse_exp = None            # 当前位置预期的内椭圆
        self.ellipse_cnn = None            # 基于CNN计算得到的内椭圆
        self.ellipse_fit = None            # 基于椭圆拟合得到的内椭圆
        self.ellipse_exp_fit = None        # 基于预期椭圆拟合得到的内椭圆
        self.ellipse_vision = None         # 基于视觉方式得到的内椭圆
        self.ellipse_kalman = None         # 基于卡尔曼滤波得到的内椭圆
        self.coil_ellipse = None           # 经过整形化的带卷内椭圆，主要用于显示

        # 跟踪误差
        self.warning = WarningType.ok
        self.err_fit_vs_exp = None         # 拟合椭圆与预期椭圆比值
        self.err_cnn_vs_exp = None         # cnn椭圆与预期椭圆比值
        self.err_cnn_vs_fit = None         # cnn椭圆与拟合椭圆比值

        # 显示轨迹跟踪
        self.show_fitting = True           # 显示椭圆拟合
        self.kalman_trace = list()          # 经过卡尔曼滤波后的轨迹
        self.vision_trace = list()          # 单纯由视觉产生的轨迹

    def init_network(self, image_size, output_size):
        network = CoilNet(img_size=image_size, output_size=output_size)
        network.add_conv_layer(channels=8, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=16, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=32, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=64, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=128, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=256, kernel_size=3, padding=1, pool_size=2)
        network.add_linear_layer_by_divider(divider=32)
        network.add_linear_layer(output_size=output_size)

        return network

    def init_kalman(self):
        self.kalman = KalmanFilter(dim_x=4, dim_z=4)
        self.kalman.F = np.array([[1.0, 1.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 1.0]])
        self.kalman.H = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])
        self.kalman.P *= 1
        self.kalman.R *= 0.1

        Q = Q_discrete_white_noise(dim=2, dt=0.3, var=0.1)
        self.kalman.Q = np.zeros((4, 4))
        self.kalman.Q[0:2, 0:2] = Q
        self.kalman.Q[2:4, 2:4] = Q

    def read_trace_stat(self, file_name):
        trace_stat = {}
        csv_reader = csv.reader(open(file_name, 'r'))
        for line in csv_reader:
            center = (int(float(line[0])), int(float(line[1])))  # 椭圆中心位置
            values = (int(float(line[2]) + 0.5), int(float(line[3]) + 0.5), int(float(line[4]) + 0.5))
            trace_stat[center] = values
        self.exp_ellipse = trace_stat

    def init_one_cycle(self, init_ellipse):
        # 初始化模板匹配
        #sub_windows_cnt = (self.diff_x_range[1] - self.diff_x_range[0]) * (self.diff_y_range[1] - self.diff_y_range[0])
        #self.coil_diff = CoilOffset(self.diff_size ** 2, sub_windows_cnt)

        self.lst_frame = None
        self.ellipse_kalman = np.array(init_ellipse).astype(np.float32)

        self.kalman_trace.clear()
        self.vision_trace.clear()

        self.err_fit_vs_exp = -1
        self.err_cnn_vs_exp = -1
        self.err_cnn_vs_fit = -1

        self.kalman.x = np.array([init_ellipse[0], 0, init_ellipse[1], 0])

    def offset_by_template(self, frame, lst_center):
        if self.lst_frame is None:
            self.lst_frame = frame
            return 0, 0

        # 从前一帧图像中截取差分检测窗口
        lst_center = (lst_center[0] + self.diff_offset[0], lst_center[1] + self.diff_offset[1])
        template = get_sub_image(self.lst_frame, lst_center, self.diff_size)
        if template is None:
            return 0, 0
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 当前检测窗口
        target = get_sub_image(frame, lst_center, self.diff_size * 2)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # 模板匹配
        result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        off_x = min_loc[0] - self.diff_size + int(self.diff_size / 2)
        off_y = min_loc[1] - self.diff_size + int(self.diff_size / 2)

        # 保存当前帧用于下一次差分计算
        self.lst_frame = frame

        return off_x, off_y

    """
    def offset_by_frame_diff(self, frame, lst_center, cur_center):
        if self.lst_frame is None:
            self.lst_frame = frame
            return 0, 0

        # 从前一帧图像中截取差分检测窗口
        lst_center = (lst_center[0] + self.diff_offset[0], lst_center[1] + self.diff_offset[1])
        sub_img = get_sub_image(self.lst_frame, lst_center, self.diff_size)
        if sub_img is None:
            return 0, 0
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

        return off_x, off_y
    """

    def ellipse_expected(self, center, neighbor=7):
        if self.exp_ellipse is None:
            return None

        long = short = angle = count = 0
        offset = int(neighbor / 2)
        for x in range(neighbor):
            for y in range(neighbor):
                key = (center[0] + x - offset, center[1] + y - offset)
                if key in self.exp_ellipse:
                    ellipse = self.exp_ellipse[key]
                    long += ellipse[0]
                    short += ellipse[1]
                    angle += ellipse[2]
                    count += 1
        if count:
            return np.array((center[0], center[1], int(long / count), int(short / count), int(angle / count)))
        else:
            return None

    def fit_ellipse(self, frame, init_ellipse):
        init_ellipse = init_ellipse.copy()
        init_center = (init_ellipse[0], init_ellipse[1])

        proc_image = get_sub_image(frame, center=init_center, img_size=self.WINDOW_SIZE)
        proc_image = edge_image(proc_image)

        init_ellipse[0] = init_ellipse[1] = int(self.WINDOW_SIZE / 2)
        proc_image = filter_around_ellipse(proc_image, init_ellipse)
        ellipse = fit_ellipse(proc_image, threshold=50)
        if ellipse is None:
            return None

        if self.show_fitting:
            img = cv2.cvtColor(proc_image, cv2.COLOR_GRAY2BGR)
            cv2.ellipse(img, (int(ellipse[0]), int(ellipse[1])), (int(ellipse[2]), int(ellipse[3])), int(ellipse[4]),
                        0, 360, (0, 255, 0), 1)
            cv2.imshow("Ellipse fitting", img)

        ellipse[0], ellipse[1] = calc_original_pos((ellipse[0], ellipse[1]), self.WINDOW_SIZE, init_center)
        # 对椭圆进行滤波，内圆中心点由卡尔曼滤波，不在这里处理
        if self.lst_ellipses is not None:
            for i in range(2, 5, 1):
                ellipse[i] = self.lst_ellipses[i] * self.low_pass_k + ellipse[i] * (1.0 - self.low_pass_k)
        self.lst_ellipses = ellipse

        return ellipse

    @staticmethod
    def get_ellipse_array(ellipse_temp):
        canvas = np.zeros((512, 512), dtype=np.uint8)
        cv2.ellipse(canvas, (256, 256), (ellipse_temp[0], ellipse_temp[1]),
                    ellipse_temp[2], ellipse_temp[3], ellipse_temp[4], 255,
                    ellipse_temp[5])
        ellipse = np.array(np.where(canvas > 0))
        return ellipse

    def fit_exp_ellipse(self, image, ellipse_temp):
        offset = 16
        max_fitness = 0
        bst_center = None

        ellipse = self.get_ellipse_array(ellipse_temp)
        max_x = np.max(ellipse[1])
        min_x = np.min(ellipse[1])
        max_y = np.max(ellipse[0])
        min_y = np.min(ellipse[0])
        for x in range(-offset, offset):
            for y in range(-offset, offset):
                if min_x + x < 0 or max_x + x >= image.shape[1] or min_y + y < 0 or max_y + y >= image.shape[0]:
                    continue

                fitness = np.sum(image[ellipse[0] + y, ellipse[1] + x] ** 2)
                if fitness > max_fitness:
                    max_fitness = fitness
                    bst_center = (x + 256, y + 256)
        if bst_center is not None:
            ellipse = (bst_center[0], bst_center[1], ellipse_temp[0], ellipse_temp[1], ellipse_temp[2])
            return ellipse
        else:
            return None

    @staticmethod
    def calc_ellipse_diff(ellipse_exp, ellipse):
        long_err = math.fabs(ellipse[2] - ellipse_exp[2]) / ellipse_exp[2]
        short_err = math.fabs(ellipse[3] - ellipse_exp[3]) / ellipse_exp[3]
        angle_err = math.fabs(ellipse[3] - ellipse_exp[3]) / 360.0

        total_err = math.sqrt(long_err**2 + short_err**2 + angle_err**2)
        return total_err

    @staticmethod
    def draw_ellipse(image, ellipse, color):
        cv2.ellipse(image, (ellipse[0], ellipse[1]), (ellipse[2], ellipse[3]), ellipse[4], 0, 360, color, 2)

    @staticmethod
    def show_ellipse(image, ellipse, color, caption):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(image, (ellipse[0], ellipse[1]), (ellipse[2], ellipse[3]), ellipse[4], 0, 360, color, 1)
        cv2.imshow(caption, image)

    def calc_coil_inner_ellipse(self, frame, init_ellipse):
        self.ellipse_exp = None
        self.ellipse_fit = None
        self.ellipse_exp_fit = None
        self.ellipse_vision = None
        self.warning = WarningType.ok

        # 基于CNN网络搜索带卷中心椭圆，由于噪音影响，该椭圆存在一定误差
        self.ellipse_cnn = self.center_by_net(frame, self.WINDOW_SIZE, init_ellipse)

        # 刚开始跟踪带卷时，精度可能较差，先不做精细拟合处理
        if self.ellipse_cnn[0] > self.noise_filter_begin:
            self.ellipse_vision = self.ellipse_cnn
            return self.ellipse_vision

        # 找出该位置的的期望椭圆，计算椭圆掩膜，用于过滤掩膜之外的边缘图像
        self.ellipse_exp = self.ellipse_expected(self.ellipse_cnn, neighbor=15)
        ellipse_mask = self.ellipse_exp if self.ellipse_exp is not None else self.ellipse_cnn

        # 截取带卷中心附近子图，计算边缘，滤除内椭圆之外的边缘
        image = get_sub_image(frame, center=ellipse_mask, img_size=self.WINDOW_SIZE)
        image = edge_image(image)
        filter_ellipse = (int(self.WINDOW_SIZE/2), int(self.WINDOW_SIZE/2), ellipse_mask[2], ellipse_mask[3], ellipse_mask[4])
        filter_range = (180, 430)  #(0, 360) if ellipse_mask[1] < 1340 else (180, 430)
        image = filter_around_ellipse(image, ellipse=filter_ellipse, range=filter_range)

        # 基于椭圆拟合计算带卷内椭圆
        self.ellipse_fit = fit_ellipse(image, threshold=50)
        if self.ellipse_fit is not None:
            self.ellipse_fit = (self.ellipse_fit + 0.5).astype(np.int32)
            if self.show_fitting:
                self.show_ellipse(image, self.ellipse_fit, (0, 255, 0), "Ellipse fitting")
            self.ellipse_fit[0], self.ellipse_fit[1] = \
                calc_original_pos(pos=self.ellipse_fit, img_size=self.WINDOW_SIZE, init_center=ellipse_mask)

        # 如果预期椭圆存在，则用该椭圆拟合带卷内椭圆
        if self.ellipse_exp is not None:
            ellipse_temp = (self.ellipse_exp[2], self.ellipse_exp[3], self.ellipse_exp[4], 0, 360, 3)
            self.ellipse_exp_fit = self.fit_exp_ellipse(image, ellipse_temp)
            if self.ellipse_exp_fit is not None:
                self.ellipse_exp_fit = np.array(self.ellipse_exp_fit)
                if self.show_fitting:
                    self.show_ellipse(image, self.ellipse_exp_fit, (0, 255, 0), "Expected ellipse fitting")
                self.ellipse_exp_fit[0], self.ellipse_exp_fit[1] = \
                    calc_original_pos(pos=self.ellipse_exp_fit, img_size=self.WINDOW_SIZE, init_center=ellipse_mask)

        # 检测拟合结果，判断是否存在显著错误
        # 如果椭圆拟合失败，说明干扰过于明显
        if self.ellipse_fit is None:
            self.warning = WarningType.ellipse_fitting_failed

        # 如果预期椭圆数据存在，则检查拟合所得椭圆和卷积所得椭圆与其的形态差异
        self.err_fit_vs_exp = -1
        self.err_cnn_vs_exp = -1
        self.err_cnn_vs_fit = -1
        if self.ellipse_exp is not None:
            self.err_cnn_vs_exp = self.calc_ellipse_diff(self.ellipse_cnn, self.ellipse_exp)
            if self.ellipse_fit is not None:
                self.err_fit_vs_exp = self.calc_ellipse_diff(self.ellipse_fit, self.ellipse_exp)
        if self.ellipse_fit is not None:
            self.err_cnn_vs_fit = self.calc_ellipse_diff(self.ellipse_cnn, self.ellipse_fit)

        # 综合结果
        if self.ellipse_fit is not None:
            if self.ellipse_exp_fit is not None:
                self.ellipse_vision = ((self.ellipse_fit + self.ellipse_exp_fit) / 2 + 0.5).astype(np.int32)
            else:
                self.ellipse_vision = self.ellipse_fit
        else:
            self.ellipse_vision = self.ellipse_cnn

        return self.ellipse_vision

    def kalman_filter(self, center, offset):
        x = np.array((center[0], offset[0], center[1], offset[1]))
        self.kalman.predict()
        self.kalman.update(x)
        return np.array((self.kalman.x[0], self.kalman.x[2]))

    def analyze_one_frame(self, frame):
        # 基于视觉方式检测带卷内椭圆
        ellipse_vision = self.calc_coil_inner_ellipse(frame, init_ellipse=self.ellipse_kalman)

        # locate coil offset by frame difference
        #ox, oy = self.offset_by_frame_diff(frame, lst_center=self.ellipse_kalman, cur_center=ellipse_vision)
        ox, oy = self.offset_by_template(frame, lst_center=self.ellipse_kalman)

        # kalman filter
        self.ellipse_kalman[0:2] = self.kalman_filter(ellipse_vision, (ox, oy))
        self.ellipse_kalman[2:5] = self.ellipse_kalman[2:5] * self.low_pass_k + \
                                   ellipse_vision[2:5] * (1.0 - self.low_pass_k)

        # 内椭圆整形化，以便于显示
        self.coil_ellipse = (self.ellipse_kalman + 0.5).astype(np.int32)

        # 记录运动轨迹，画图用
        self.kalman_trace.append(self.coil_ellipse)
        self.vision_trace.append(self.ellipse_vision)

        return self.coil_ellipse

    def show_trace(self, frame):
        # show coil center obtained by vision and kalman filter
        if self.ellipse_vision is not None:
            cv2.circle(frame, (self.ellipse_vision[0], self.ellipse_vision[1]), 2, (255, 0, 0), 2)
        cv2.circle(frame, (self.coil_ellipse[0], self.coil_ellipse[1]), 2, (0, 255, 0), 2)

        self.draw_ellipse(frame, self.coil_ellipse, color=(0, 255, 0))

        # show center moving trace
        lst_ellipse = self.kalman_trace[0]
        for _, ellipse in enumerate(self.kalman_trace):
                cv2.line(frame, lst_ellipse[0:2], ellipse[0:2], (0, 255, 0), 2)
                lst_ellipse = ellipse

        # 显示差分计算区域
        diff_center = (self.ellipse_kalman[0] + self.diff_offset[0], self.ellipse_kalman[1] + self.diff_offset[1])
        cv2.rectangle(frame,
                      (int(diff_center[0] - self.diff_size / 2), int(diff_center[1] - self.diff_size / 2)),
                      (int(diff_center[0] + self.diff_size / 2), int(diff_center[1] + self.diff_size / 2)),
                      (0, 0, 255), 2)


if __name__ == '__main__':
    Net = CenterTracer().coil_net
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