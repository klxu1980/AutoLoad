import torch
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import csv
import time
from enum import Enum
from CoilNet import CoilNet
from ImageSet import LabelType
from ImageSet import ImageSet
from CoilLocate import CoilLocator
from CoilStatus import CoilExist
from CoilImageProc import *
from FixedParams import *

"""
filterpy需要单独安装，执行指令"pip install filterpy"
VC调用filterpy时，有可能会出现"DLL load failed while importing _arpack"错误。该错误通常是scipy的版本兼容性问题引起的。
可以先删除，再安装scipy。具体方法为：在anaconda prompt下面，执行pip uninstall scipy，再执行pip install scipy。
"""


class WarningType(Enum):
    ok = 0
    ellipse_fitting_failed = 1
    ellipse_large_error = 2


class TrackingStatus(Enum):
    NO_COIL  = 0,         # 没有检测到带卷
    LOCATING = 1,         # 正在搜索带卷初始位置
    TRACKING = 2          # 正在跟踪


class CenterTracer(object):
    def __init__(self):
        self.WINDOW_SIZE = 512

        self.output_size = LabelType.InnerOnly
        self.coil_net = self.__init_network(self.WINDOW_SIZE, self.output_size)
        self.coil_net.init_model()

        # 使用CNN确定中心位置时，在初始位置附近，随机产生一系列窗口，对窗口中的图像检测中心，然后求平均
        self.net_x_range = (-20, 20)        # 随机窗口x方向范围
        self.net_y_range = (-20, 20)        # 随机窗口y方向范围
        self.net_count = CNN_SUB_IMAGE_CNT  # 随机窗口数量

        # 通过帧差分检测偏移量时，需要确定检测窗口大小，和窗口相对初始中心点的偏移位置
        self.diff_size = 256           # 检测窗口尺寸，必须是2的指数
        self.diff_offset = (180, -180)
        self.lst_frame = None

        # 带卷是否存在，和带卷初始位置检测
        self.coil_exist = CoilExist(src_size=1000, src_position=(1550, 0))
        self.coil_exist.init_model()

        self.coil_locator = CoilLocator()

        # 带卷小车在受控运动时的速度(像素/秒)，由之前的记录统计获得(CarAnalyzer.py)
        self.car_spd_up           = (-4.76538781, 11.23054992)
        self.car_spd_down         = (0.03374108, 29.97134124)
        self.car_spd_forward_top  = (-32.10138965, 20.3808698 )
        self.car_spd_backward_top = (30.50859806, -18.19230394)
        self.car_spd_forward_btn  = (-19.94236851, 20.00267209)
        self.car_spd_backward_btn = (np.nan, np.nan)
        self.car_confidence = 0.65      # 基于小车速度计算出来的位移量的置信度
        self.car_considered = True      # 计算位移量时，是否考虑小车速度信息

        # 初始化卡尔曼滤波器
        self.__init_kalman()

        # 计时器
        self.start_time = time.perf_counter()
        self.proc_interval = 0.0
        self.tracking_frame_id = 0         # 跟踪开始后的当前帧号

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
        self.img_fitting = None
        self.img_exp_fitting = None
        self.exp_fit_confidence = 0.75

        # 跟踪误差
        self.warning = WarningType.ok
        self.err_fit_vs_exp = -1         # 拟合椭圆与预期椭圆比值
        self.err_cnn_vs_exp = -1         # cnn椭圆与预期椭圆比值
        self.err_cnn_vs_fit = -1         # cnn椭圆与拟合椭圆比值
        self.err_ellipse    = -1         # 最终椭圆检测结果误差
        self.err_movement   = -1
        self.coil_within_route = True    # 带卷跟踪超出合理范围

        # 带卷跟踪状态
        self.tracking_status = TrackingStatus.NO_COIL

        # 显示轨迹跟踪
        self.kalman_trace = list()          # 经过卡尔曼滤波后的轨迹
        self.vision_trace = list()          # 单纯由视觉产生的轨迹

        # 各个主要功能的计算时间
        self.edge_time = 0
        self.cnn_time = 0         # CNN计算椭圆耗时
        self.fit_time = 0         # 椭圆拟合耗时
        self.fit_exp_time = 0     # 椭圆模板拟合耗时
        self.offset_time = 0      # 帧之间偏移量计算时间
        self.analyze_time = 0

        offset = 16
        self.x_offset = np.repeat(np.arange(-offset, offset, 1), offset * 2)
        self.y_offset = np.tile(np.arange(-offset, offset, 1), offset * 2)

    @staticmethod
    def __init_network(image_size, output_size):
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

    def __init_kalman(self):
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

    def __init_one_cycle(self, init_ellipse):
        self.lst_frame = None
        self.ellipse_kalman = np.array(init_ellipse).astype(np.float32)
        self.start_time = time.perf_counter()

        self.kalman_trace.clear()
        self.vision_trace.clear()
        self.coil_ellipse = None

        self.kalman.x = np.array([init_ellipse[0], 0, init_ellipse[1], 0])

    def __center_by_net(self, frame, img_size, init_center):
        begin_time = time.perf_counter()

        # 预先提取图像边缘以提高精度
        # 为了减小计算耗时，仅对目标附近的区域进行边缘提取
        proc_image = get_sub_image(frame, center=init_center, img_size=img_size + 100)

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

        # 计算目标在整个图像中的位置
        values[0] += init_center[0] - sub_center[0]
        values[1] += init_center[1] - sub_center[1]

        # self.cnn_time = int((time.perf_counter() - begin_time) * 1000)
        return values

    def __offset_by_template(self, frame, lst_center):
        begin_time = time.perf_counter()

        if self.lst_frame is None:
            self.lst_frame = frame
            return 0, 0

        # 从前一帧图像中截取差分检测窗口
        lst_center = (lst_center[0] + self.diff_offset[0], lst_center[1] + self.diff_offset[1])
        template = get_sub_image(self.lst_frame, lst_center, self.diff_size)
        if template is None:
            return 0, 0

        # 当前检测窗口
        target = get_sub_image(frame, lst_center, self.diff_size * 2)

        # 模板匹配
        result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        off_x = min_loc[0] - self.diff_size + int(self.diff_size / 2)
        off_y = min_loc[1] - self.diff_size + int(self.diff_size / 2)

        # 保存当前帧用于下一次差分计算
        self.lst_frame = frame

        self.offset_time = int((time.perf_counter() - begin_time) * 1000)
        return off_x, off_y

    def __coil_on_top(self, coil_ellipse):
        # 判断带卷是在高处还是基坑中
        return coil_ellipse[1] < 1300

    def __offset_by_car_movement(self, coil_ellipse, plc, interval):
        # 应当考虑小车的惯性，建立一个一节惯性模型
        if plc.car_up:
            spd = self.car_spd_up
        elif plc.car_down:
            spd = self.car_spd_down
        elif plc.car_forward:
            spd = self.car_spd_forward_top if self.__coil_on_top(coil_ellipse) else self.car_spd_forward_btn
        elif plc.car_backward:
            spd = self.car_spd_backward_top if self.__coil_on_top(coil_ellipse) else self.car_spd_backward_btn
        else:
            spd = (0.0, 0.0)

        return spd[0] * interval, spd[1] * interval

    def __ellipse_expected(self, center, neighbor=7):
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

    @staticmethod
    def __get_ellipse_array(temp_size, ellipse_temp):
        """
        将椭圆每个像点的坐标转换到数组里
        这部分程序以后可以做一下优化
        :param ellipse_temp:
        :return:
        """
        canvas = np.zeros((temp_size, temp_size), dtype=np.uint8)
        cv2.ellipse(canvas, (int(temp_size / 2), int(temp_size / 2)), (ellipse_temp[0], ellipse_temp[1]),
                    ellipse_temp[2], ellipse_temp[3], ellipse_temp[4], 255,
                    ellipse_temp[5])
        ellipse = np.array(np.where(canvas > 0))
        return ellipse

    def __fit_exp_ellipse_fast(self, image, ellipse_temp):
        # 目前测试看，这个拟合算法不如之前的快
        offset = 16
        row_cnt = offset * offset * 4

        ellipse = self.__get_ellipse_array(temp_size=image.shape[0], ellipse_temp=ellipse_temp)
        x_idx = np.tile(ellipse[1], row_cnt).reshape((row_cnt, ellipse.shape[1]))
        y_idx = np.tile(ellipse[0], row_cnt).reshape((row_cnt, ellipse.shape[1]))

        x_idx += self.x_offset
        y_idx += self.y_offset

        img = image[y_idx, x_idx]
        row_norm = np.linalg.norm(img, ord=2, axis=0)
        max_row = np.argmax(row_norm)

        x = self.x_offset[max_row]
        y = self.y_offset[max_row]

        ellipse = (x + int(image.shape[0] / 2), y + int(image.shape[1] / 2), ellipse_temp[0], ellipse_temp[1], ellipse_temp[2])
        return ellipse

    def __fit_exp_ellipse(self, image, ellipse_temp):
        """
        找出预期椭圆最接近的位置
        :param image:
        :param ellipse_temp:
        :return:
        """
        offset = 16
        max_fitness = 0
        bst_center = None

        begin_time = time.perf_counter()
        ellipse = self.__get_ellipse_array(temp_size=image.shape[0], ellipse_temp=ellipse_temp)
        for x in range(-offset, offset):
            for y in range(-offset, offset):
                fitness = np.sum(image[ellipse[0] + y, ellipse[1] + x] ** 2)
                if fitness > max_fitness:
                    max_fitness = fitness
                    bst_center = (x + int(image.shape[0] / 2), y + int(image.shape[0] / 2))
        if bst_center is not None:
            ellipse = (bst_center[0], bst_center[1], ellipse_temp[0], ellipse_temp[1], ellipse_temp[2])
            # self.fit_exp_time = int((time.perf_counter() - begin_time) * 1000)
            return ellipse
        else:
            return None

    @staticmethod
    def __calc_ellipse_diff(ellipse_exp, ellipse):
        # 角度误差的量纲和长度不一样，混合在一起比较麻烦，因此不考虑角度误差
        long_err = (ellipse[2] - ellipse_exp[2]) / ellipse_exp[2]
        short_err = (ellipse[3] - ellipse_exp[3]) / ellipse_exp[3]
        total_err = math.sqrt(long_err**2 + short_err**2)
        return total_err

    @staticmethod
    def __draw_ellipse(image, ellipse, color):
        cv2.ellipse(image, (ellipse[0], ellipse[1]), (ellipse[2], ellipse[3]), ellipse[4], 0, 360, color, 2)

    @staticmethod
    def __show_ellipse(image, ellipse, color, caption):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(image, (ellipse[0], ellipse[1]), (ellipse[2], ellipse[3]), ellipse[4],
                    ELLIPSE_ANGLE_RANGE[0] - ellipse[4], ELLIPSE_ANGLE_RANGE[1] - ellipse[4], color, 1)
        # cv2.imshow(caption, image)
        return image

    def __calc_coil_inner_ellipse(self, frame, init_ellipse):
        """
        基于视觉检测带卷内椭圆
        :param frame:
        :param init_ellipse:
        :return:
        """
        # 注意：这里存在重复计算图像边缘的问题，需要优化
        self.ellipse_exp = None
        self.ellipse_fit = None
        self.ellipse_exp_fit = None
        self.ellipse_vision = None
        self.warning = WarningType.ok

        # 对带卷附近区域做边缘提取
        begin_time = time.perf_counter()
        sub_img_size = self.WINDOW_SIZE + 50
        sub_img = get_sub_image(frame, center=init_ellipse, img_size=sub_img_size)
        sub_img_cpy = sub_img.copy()
        embed_sub_image(img=frame, sub_img=edge_image(sub_img), center=init_ellipse)
        self.edge_time = int((time.perf_counter() - begin_time) * 1000)

        # 基于CNN网络搜索带卷中心椭圆，由于噪音影响，该椭圆存在一定误差
        begin_time = time.perf_counter()
        self.ellipse_cnn = self.__center_by_net(frame, self.WINDOW_SIZE, init_ellipse)
        self.cnn_time = int((time.perf_counter() - begin_time) * 1000)

        # 刚开始跟踪带卷时，精度可能较差，先不做精细拟合处理
        if self.ellipse_cnn[0] > self.noise_filter_begin:
            self.ellipse_vision = self.ellipse_cnn
            return self.ellipse_vision

        # 找出该位置的的期望椭圆，计算椭圆掩膜，
        begin_time = time.perf_counter()
        self.ellipse_exp = self.__ellipse_expected(self.ellipse_cnn, neighbor=15)
        ellipse_mask = self.ellipse_exp if self.ellipse_exp is not None else self.ellipse_cnn

        # 截取带卷中心附近子图，计算边缘，滤除内椭圆之外的边缘
        image = get_sub_image(frame, center=ellipse_mask, img_size=sub_img_size)
        filter_ellipse = (sub_img_size // 2, sub_img_size // 2, ellipse_mask[2], ellipse_mask[3], ellipse_mask[4])
        image = filter_around_ellipse(image, ellipse=filter_ellipse, range=ELLIPSE_ANGLE_RANGE)

        # 基于椭圆拟合计算带卷内椭圆
        self.img_fitting = None
        self.img_exp_fitting = None
        begin_time = time.perf_counter()
        self.ellipse_fit = fit_ellipse(image, threshold=50)
        self.fit_time = (int(time.perf_counter() - begin_time) * 1000)
        if self.ellipse_fit is not None:
            self.ellipse_fit = (self.ellipse_fit + 0.5).astype(np.int32)
            if SHOW_FITTING_ELLIPSE:
                self.img_fitting = self.__show_ellipse(image, self.ellipse_fit, (0, 255, 0), "Ellipse fitting")
            self.ellipse_fit[0], self.ellipse_fit[1] = \
                calc_original_pos(pos=self.ellipse_fit, img_size=sub_img_size, init_center=ellipse_mask)
        self.fit_time = int((time.perf_counter() - begin_time) * 1000)

        # 如果预期椭圆存在，则用该椭圆拟合带卷内椭圆
        begin_time = time.perf_counter()
        if self.ellipse_exp is not None:
            ellipse_temp = (self.ellipse_exp[2], self.ellipse_exp[3], self.ellipse_exp[4],
                            ELLIPSE_ANGLE_RANGE[0] - self.ellipse_exp[4], ELLIPSE_ANGLE_RANGE[1] - self.ellipse_exp[4], 3)
            self.ellipse_exp_fit = self.__fit_exp_ellipse(image, ellipse_temp)
            if self.ellipse_exp_fit is not None:
                self.ellipse_exp_fit = np.array(self.ellipse_exp_fit)
                if SHOW_FITTING_ELLIPSE:
                    self.img_exp_fitting = self.__show_ellipse(image, self.ellipse_exp_fit, (0, 255, 0),
                                                               "Expected ellipse fitting")
                # print(self.ellipse_exp_fit)
                self.ellipse_exp_fit[0], self.ellipse_exp_fit[1] = \
                    calc_original_pos(pos=self.ellipse_exp_fit, img_size=sub_img_size, init_center=ellipse_mask)

        # 检测拟合结果，判断是否存在显著错误
        # 如果椭圆拟合失败，说明干扰过于明显
        if self.ellipse_fit is None:
            self.warning = WarningType.ellipse_fitting_failed

        # 恢复原始图像
        embed_sub_image(img=frame, sub_img=sub_img_cpy, center=init_ellipse)
        self.fit_exp_time = int((time.perf_counter() - begin_time) * 1000)

        # 综合结果
        # 如果椭圆拟合失败，则采用CNN检测结果
        # 如果椭圆拟合成功，且用预期椭圆拟合也成功，则对两个椭圆求平均得到结果
        # 如果椭圆拟合成功，但预期椭圆拟合失败，则用拟合结果
        if self.ellipse_fit is None:
            self.ellipse_vision = self.ellipse_cnn
        elif self.ellipse_exp_fit is None:
            self.ellipse_vision = self.ellipse_fit
        else:
            ellipse_vision = self.ellipse_exp_fit * self.exp_fit_confidence + \
                             self.ellipse_fit * (1.0 - self.exp_fit_confidence)
            self.ellipse_vision = ellipse_vision.astype(np.int32)
        return self.ellipse_vision

    def __kalman_filter(self, center, offset):
        x = np.array((center[0], offset[0], center[1], offset[1]))
        self.kalman.predict()
        self.kalman.update(x)
        return np.array((self.kalman.x[0], self.kalman.x[2]))

    def __track_coil(self, frame, plc, proc_interval):
        # 基于视觉方式检测带卷内椭圆
        ellipse_vision = self.__calc_coil_inner_ellipse(frame, init_ellipse=self.ellipse_kalman)

        # locate coil offset by frame difference
        ox, oy = self.__offset_by_template(frame, lst_center=self.ellipse_kalman)

        # 可以读取到PLC状态时，利用小车的当前运动方向修正计算带卷的运动偏移量
        if plc is not None and self.car_considered:
            ox_car, oy_car = self.__offset_by_car_movement(ellipse_vision, plc, proc_interval)
            if ox_car != np.nan:
                ox = ox_car * self.car_confidence + ox * (1.0 - self.car_confidence)
                oy = oy_car * self.car_confidence + oy * (1.0 - self.car_confidence)

        # kalman filter
        self.ellipse_kalman[0:2] = self.__kalman_filter(ellipse_vision, (ox, oy))
        self.ellipse_kalman[2:5] = self.ellipse_kalman[2:5] * self.low_pass_k + \
                                   ellipse_vision[2:5] * (1.0 - self.low_pass_k)

        # 内椭圆整形化，以便于显示
        coil_ellipse = ellipse_vision if NO_KALMAN else (self.ellipse_kalman + 0.5).astype(np.int32)
        coil_movement = None
        if self.coil_ellipse is not None:
            coil_movement = coil_ellipse[0:5] - self.coil_ellipse[0:5]
        self.coil_ellipse = coil_ellipse

        # 评估检测跟踪误差
        self.__evaluate_tracking(self.coil_ellipse, coil_movement, plc, proc_interval)

        # 记录运动轨迹，画图用
        self.kalman_trace.append(self.coil_ellipse)
        self.vision_trace.append(self.ellipse_vision)
        if len(self.kalman_trace) > 100:
            self.kalman_trace.pop(0)
        if len(self.vision_trace) > 100:
            self.vision_trace.pop(0)

    @staticmethod
    def __calc_supposed_y(begin, end, x):
        k = (end[1] - begin[1]) / (end[0] - begin[0])
        y = (x - begin[0]) * k + begin[1]
        return y

    @staticmethod
    def __calc_supposed_x(begin, end, y):
        k = (end[0] - begin[0]) / (end[1] - begin[1])
        x = (y - begin[1]) * k + begin[0]
        return x

    def __ellipse_within_bound(self, center):
        supp_y_top = self.__calc_supposed_y(coil_trans_begin, coil_fall_begin, center[0])    # 带卷在基坑上端平移时的预期高度
        supp_y_btm = self.__calc_supposed_y(alignment_keypoint, end_keypoint, center[0])     # 带卷在基坑下端平移时的预期高度
        supp_x_fall = self.__calc_supposed_x(coil_fall_begin, alignment_keypoint, center[1]) # 带卷在垂直下落时的预期水平位置
        max_err_y = 100
        max_err_x = 100

        if supp_y_top - max_err_y < center[1] < supp_y_top + max_err_y:
            return center[0] > supp_x_fall - max_err_x
        elif supp_y_btm - max_err_y < center[1] < supp_y_btm + max_err_y:
            return center[0] < supp_x_fall + max_err_x
        elif supp_x_fall - max_err_x < center[0] < supp_x_fall + max_err_x:
            return supp_y_top - max_err_y < center[1] < supp_y_btm + max_err_y
        else:
            return False

    def __evaluate_tracking(self, ellipse, movement, plc, proc_interval):
        """
        评估跟踪质量
        """
        # 如果预期椭圆数据存在，则检查拟合所得椭圆和卷积所得椭圆与其的形态差异
        if self.ellipse_exp is None:
            self.err_ellipse = np.nan
        else:
            self.err_ellipse = self.__calc_ellipse_diff(ellipse, self.ellipse_exp)

        # 评估运动量是否出现明显错误
        if plc is not None and movement is not None:
            exp_ox, exp_oy = self.__offset_by_car_movement(ellipse, plc, proc_interval)
            if exp_ox != np.nan:
                err_x = (movement[0] - exp_ox) / (exp_ox * 1.5 if math.fabs(exp_ox) > 1 else 5)
                err_y = (movement[1] - exp_oy) / (exp_oy * 1.5 if math.fabs(exp_oy) > 1 else 5)
                self.err_movement = math.sqrt(err_x ** 2 + err_y ** 2)

    def load_model(self, model_file):
        """
        加载网络模型
        :param model_file:
        :return:
        """
        self.coil_net.load_model(model_file)

    def read_trace_stat(self, file_name):
        """
        读取带卷统计参数
        :param file_name:
        :return:
        """
        trace_stat = {}
        csv_reader = csv.reader(open(file_name, 'r'))
        for line in csv_reader:
            center = (int(float(line[0])), int(float(line[1])))  # 椭圆中心位置
            values = (int(float(line[2]) + 0.5), int(float(line[3]) + 0.5), int(float(line[4]) + 0.5))
            trace_stat[center] = values
        self.exp_ellipse = trace_stat

    def restart_tracking(self):
        """
        重新启动跟踪
        :return:
        """
        self.tracking_status = TrackingStatus.NO_COIL
        self.tracking_frame_id = 0

    def analyze_one_frame(self, frame, plc=None):
        """
        分析视频的一帧画面，检测和跟踪带卷。检测跟踪有3个状态：
        1. NO_COIL: 初始状态下，默认为堆料区没有带卷，首先检测是否存在带卷(self.coil_exist)。如果检测出带卷，则进入下一状态
        2. LOCATING: 由self.coil_locator检测带卷的初始位置，如果初始位置满足跟踪条件，则进入跟踪状态
        3. TRACKING: 对带卷进行持续精确跟踪定位。上一级程序(AutoLoad)使用跟踪数据控制上卷(自动控制时)，或者监控上卷阶段(监控模式)。
                     当上卷结束时，上层程序调用restart_tracking()函数重新开始新的带卷跟踪。如果跟踪位置时超出预计轨迹范围，
                     则置coil_within_route=False，通知上层程序，由上层程序决定是否重新开始跟踪。
        :param frame:
        :param plc:
        :return:
        """
        # 采样时间间隔
        self.proc_interval = time.perf_counter() - self.start_time
        self.start_time = time.perf_counter()

        # 将各个主要功能耗时初始化为-1
        self.edge_time = -1
        self.offset_time = -1
        self.cnn_time = -1
        self.fit_time = -1
        self.fit_exp_time = -1
        self.analyze_time = -1

        # 分析图像
        begin_time = time.perf_counter()
        if self.tracking_status == TrackingStatus.NO_COIL:
            if self.coil_exist.analyze_one_frame(frame):
                self.tracking_status = TrackingStatus.LOCATING
            self.coil_locator.init_one_cycle()

        if self.tracking_status == TrackingStatus.LOCATING:
            center = self.coil_locator.analyze_one_frame(frame)
            if TRACKING_BEGIN_X < center[0]:   # < coil_trans_begin[0]:   # 当带卷初始位置在堆料区时，初始化带卷跟踪
                self.__init_one_cycle(init_ellipse=(center[0], center[1], 100, 100, 0))
                self.tracking_status = TrackingStatus.TRACKING
                self.tracking_frame_id = 0

        if self.tracking_status == TrackingStatus.TRACKING:
            # 跟踪带卷，检查带卷是否超出合理范围
            # 刚刚由定位阶段转到跟踪阶段时，跟踪误差较大，因此最开始的几帧图像不进行判断
            self.tracking_frame_id += 1
            self.__track_coil(frame, plc, self.proc_interval)
            if self.tracking_frame_id < 5:
                self.coil_within_route = True
            else:
                self.coil_within_route = self.__ellipse_within_bound(self.coil_ellipse)

        self.analyze_time = int((time.perf_counter() - begin_time) * 1000)
        return self.coil_ellipse, self.tracking_status

    def show_trace(self, frame):
        if self.coil_ellipse is None:
            return frame

        # show coil center obtained by vision and kalman filter
        if self.ellipse_vision is not None:
            cv2.circle(frame, (self.ellipse_vision[0], self.ellipse_vision[1]), 2, (255, 0, 0), 2)
        cv2.circle(frame, (self.coil_ellipse[0], self.coil_ellipse[1]), 2, (0, 255, 0), 2)
        self.__draw_ellipse(frame, self.coil_ellipse, color=(0, 255, 0))

        # show center moving trace
        if len(self.kalman_trace) > 0:
            lst_ellipse = self.kalman_trace[0]
            for _, ellipse in enumerate(self.kalman_trace):
                p1 = (lst_ellipse[0], lst_ellipse[1])
                p2 = (ellipse[0], ellipse[1])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                lst_ellipse = ellipse

        # 显示差分计算区域
        # diff_center = (self.ellipse_kalman[0] + self.diff_offset[0], self.ellipse_kalman[1] + self.diff_offset[1])
        # cv2.rectangle(frame,
        #               (int(diff_center[0] - self.diff_size / 2), int(diff_center[1] - self.diff_size / 2)),
        #               (int(diff_center[0] + self.diff_size / 2), int(diff_center[1] + self.diff_size / 2)),
        #               (0, 0, 255), 2)


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
        train_set = ImageSet(output_size=LabelType.InnerOnly, img_size=512)
        train_set.load("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\Train_no_noise")
        eval_set = train_set.random_subset(sample_ration=0.3)

        train_set.random_noise = False
        train_set.batch_size = 1000
        eval_set.random_noise = False
        eval_set.batch_size = 1000
        Net.train(train_set, eval_set, epoch_cnt=10000, mini_batch=64)
    else:
        Net.load_model("带卷定位2022-08-06-22-02.pt")
        eval_set = ImageSet(output_size=LabelType.InnerOnly, img_size=512)
        eval_set.load("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\Eval_no_noise")
        eval_set.random_noise = False
        Net.test_net(eval_set)
        #Net.denoise_test(eval_set)