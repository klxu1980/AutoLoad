import cv2
import time
import os
import glob
import datetime
import csv
import numpy as np
from enum import Enum

from CenterTracer import CenterTracer
from CoilLocate import CoilLocator
from CoilStatus import CoilOpenStatus
from CoilStatus import CoilPosStatus
from CoilStatus import CoilExist
from PLCComm import PLCComm
from HKCamera import HKCamera
from CoilImageProc import *
from FilePath import PYTHON_PATH

"""
从张方远程序中拿出来的参数
"""
PositionL1P1 = (1000, 950)   # 定位线1点1
PositionL1P2 = (1230, 810)   # 定位线1点2
PositionL2P1 = (1230, 810)   # 定位线2点1
PositionL2P2 = (2280, 1260)  # 定位线2点2
PositionL3P1 = (2280, 1260)  # 定位线3点1
PositionL3P2 = (1880, 1950)  # 定位线3点2

start_keypoint = (1900, 500)  #全局点:预测大卷起始点
start_keypoint1 = (1900, 530) #全局点:预测小卷起始点

drop_keypoint = (1400,800)   #全局点：大卷关键点下降点
drop_keypoint1 = (1400, 830) #全局点：小卷关键点下降点

alignment_keypoint = (1420, 1510) #全局点：关键点对准点
end_keypoint = (1210, 1710)		  #全局点：预测结束点

stage1_up1 =(1900, 460)   # 平移阶段1运动上区间线右点
stage1_up2 = (1365, 781)   # 平移阶段1运动上区间线左点
stage1_down1 = (1910, 585) # 平移阶段1运动下区间线右点
stage1_down2 = (1440, 867) # 平移阶段1运动下区间线左点

stage2_left1 = stage1_up2	# 下降阶段2运动左区间线上点
stage2_left2 = (1385, 1490)     # 下降阶段2运动左区间线下点
stage2_right1 = stage1_down2 # 下降阶段2运动右区间线上点
stage2_right2 = (1455, 1530)    # 下降阶段2运动右区间线下点

stage3_up1 = stage2_left2    # 平移阶段3运动上区间线右点
stage3_up2 = (1200, 1685)	    # 平移阶段3运动上区间线左点
stage3_down1 = stage2_right2 # 平移阶段3运动下区间线右点
stage3_down2 = (1240, 1735)		# 平移阶段3运动下区间线左点


INIT_COIL_CENTER = (1800, 630)   # 带卷在堆料区的初始位置。通常会首先进行带卷的初始位置定位，如果没有执行，则默认使用该位置。
AIMING_BEGIN = (1450, 1450)      # 带卷对准开卷机的检测开始和停止位置
AIMING_END = end_keypoint[0]

SCREEN_WIDTH = 1900
SHOW_EDGE_IN_TEST = False


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


class LoadStage(Enum):
    WAITING_4_LOADING  = 0,   # 在堆料区等待上料
    MOVING_TO_UNPACKER = 1,   # 向基坑中移动
    INTO_UNPACKER      = 2,   # 进入开卷机(判断是否到达开卷位置)
    UNPACKING          = 3,   # 开卷(判断是否开卷)
    UPLOAD_COMPLETED   = 4    # 开卷完成


class AutoLoader(object):
    def __init__(self, demo_ctrl=True, auto_ctrl=False):
        self.demo_ctrl = demo_ctrl
        self.auto_ctrl = auto_ctrl

        # 加载各个检测模型
        self.__load_models()

        # 初始化实际控制使用的硬件
        # 如果硬件打开失败，则退出
        if not demo_ctrl:
            if not self.__init_hardwares():
                exit(0)

        # 以下变量主要在demo中使用到
        self.loading_stage = LoadStage.WAITING_4_LOADING     # 自动上卷的各个阶段
        self.proc_time = 0.0                                 # 自动上卷算法处理时间

        # 试验测试中，如果需要存储每个视频的带卷轨迹数据，则设置save_trace=True
        self.save_trace = False
        self.csv = None

        # 保存带卷轨迹跟踪视频
        self.default_saving_path = "e:\\NewTest\\"
        self.saving_video = False
        self.video_save = None
        self.paused = False

        self.video_raw = None
        self.video_proc = None

        # 带卷对准开卷机时的轨迹线
        self.aiming_ok_line = None
        self.aiming_low_line = None
        self.aiming_high_line = None
        self.load_aiming_lines()

    def __load_models(self):
        # self.coil_tracer是核心跟踪器
        self.coil_tracer = CenterTracer()
        self.coil_tracer.load_model(PYTHON_PATH + "带卷定位_有干扰_2022-08-05-16-05.pt")

        # 带卷初始位置检测器
        self.coil_locator = CoilLocator()
        self.coil_locator.load_model(PYTHON_PATH + "带卷初始位置2022-08-18-22-55.pt")

        # 上卷检测器
        self.coil_pos_status = CoilPosStatus(src_size=1200, src_position=(0, 800))
        self.coil_pos_status.init_model()
        self.coil_pos_status.load_model(PYTHON_PATH + "上卷检测2020-10-24-16-30.pt")

        # 开卷检测器
        self.coil_open_status = CoilOpenStatus(src_size=1200, src_position=(0, 800))
        self.coil_open_status.init_model()
        self.coil_open_status.load_model(PYTHON_PATH + "开卷检测2020-10-24-12-12.pt")

        # 鞍座检测器，判断带卷是否在鞍座上
        self.coil_exist_status = CoilExist(src_size=1000, src_position=(1550, 0))
        self.coil_exist_status.init_model()
        self.coil_exist_status.load_model(PYTHON_PATH + "带卷存在2022-05-25-16-16.pt")

    def __init_hardwares(self):
        # 与PLC的TCP通信
        self.plc = PLCComm()
        if not self.plc.connect():
            return False

        # 打开相机
        try:
            self.camera = HKCamera()
        except Exception as e:
            print(e)
            return False

        return True

    @staticmethod
    def datetime_string():
        now = datetime.datetime.now()
        datetime_str = "%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        return datetime_str

    @staticmethod
    def show_text(image, text, pos, color=(0, 255, 0)):
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    def load_aiming_lines(self):
        csv_reader = csv.reader(open(PYTHON_PATH + "trace_stat_正常.csv", 'r'))
        for i, line in enumerate(csv_reader):
            if i == 0:
                self.aiming_ok_line = (float(line[0]), float(line[1]))
                break

        csv_reader = csv.reader(open(PYTHON_PATH + "trace_stat_偏高.csv", 'r'))
        for i, line in enumerate(csv_reader):
            if i == 1:
                self.aiming_high_line = (float(line[0]), float(line[1]))
                break

        csv_reader = csv.reader(open(PYTHON_PATH + "trace_stat_偏低.csv", 'r'))
        for i, line in enumerate(csv_reader):
            if i == 2:
                self.aiming_low_line = (float(line[0]), float(line[1]))
                break

    def save_video(self, frame):
        if self.saving_video:
            if self.video_save is None:
                datetime_str = self.datetime_string()
                size = (frame.shape[1], frame.shape[0])
                self.video_save = cv2.VideoWriter(self.default_saving_path + datetime_str + ".mp4",
                                                  cv2.VideoWriter_fourcc(*'MP4V'), 10, size)
            self.video_save.write(frame)
            self.show_text(frame, "recording video", pos=(600, 40), color=(0, 0, 255))
        else:
            self.video_save = None

    def init_record(self, frame):
        file_path_name = self.default_saving_path + self.datetime_string()

        csv_file = open(file_path_name + ".csv", "w", newline='')
        self.csv = csv.writer(csv_file)

        size = (frame.shape[1], frame.shape[0])
        self.video_raw = cv2.VideoWriter(file_path_name + "-original.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 5, size)
        self.video_proc = cv2.VideoWriter(file_path_name + "-processed.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 5, size)

    def record_raw_video(self, frame):
        self.video_raw.write(frame)

    def record_proc_video(self, frame):
        self.video_proc.write(frame)

    def save_ellipse(self, ellipse):
        if ellipse is not None:
            row = ["%d" % value for value in ellipse]
            time_str = "%1.2f" % self.coil_tracer.proc_interval
            row.append(time_str)
            self.csv.writerow(row)

    def show_key_operations(self, frame, row_begin):
        row = row_begin
        self.show_text(frame, "P(p): pause", (20, row))
        row += 60
        self.show_text(frame, "->: move forward", (20, row))
        row += 60
        self.show_text(frame, "<-: move backward", (20, row))
        row += 60
        self.show_text(frame, "S(s): start tracing and recording", (20, row))
        row += 60
        self.show_text(frame, "E(e): stop tracing and recording", (20, row))
        row += 60
        self.show_text(frame, "R(r): save current picture", (20, row))
        row += 60
        self.show_text(frame, "L(l): detect if the coil is loaded", (20, row))
        row += 60
        self.show_text(frame, "O(o): detect if the coil is opened", (20, row))
        row += 60
        self.show_text(frame, "g(G): show coil edge", (20, row))
        row += 60
        return row

    def draw_aiming_lines(self, image):
        x_begin = 1200
        x_end = 1500
        cv2.line(image,
                 (x_begin, int(x_begin * self.aiming_high_line[0] + self.aiming_high_line[1])),
                 (x_end, int(x_end * self.aiming_high_line[0] + self.aiming_high_line[1])),
                 color=(0, 0, 255), thickness=2)
        cv2.line(image,
                 (x_begin, int(x_begin * self.aiming_low_line[0] + self.aiming_low_line[1])),
                 (x_end, int(x_end * self.aiming_low_line[0] + self.aiming_low_line[1])),
                 color=(0, 0, 255), thickness=2)

    def check_aiming_status(self, center):
        lower_bound = center[0] * self.aiming_low_line[0] + self.aiming_low_line[1]
        upper_bound = center[0] * self.aiming_high_line[0] + self.aiming_high_line[1]
        if center[1] > lower_bound:
            return 1
        elif center[1] < upper_bound:
            return 2
        else:
            return 0

    def init_one_cycle(self):
        """
        准备开始新带卷的自动上卷控制。只需要上卷时调用一次，也可多次调用，对上卷无影响。
        :return: 无
        """
        init_ellipse = (INIT_COIL_CENTER[0], INIT_COIL_CENTER[1], 0, 0, 0)
        self.coil_locator.init_one_cycle()
        self.coil_tracer.init_one_cycle(init_ellipse=init_ellipse)
        self.loading_stage = LoadStage.WAITING_4_LOADING

    def wait_for_loading(self, frame):
        """
        带卷位于堆料区等待上卷，持续定位其初始坐标(并滤波)，将初始位置写入跟踪器初始位置中
        :param frame: 由摄像头获得的带卷RGB图像
        :return: 带卷的初始位置，是一个(int, int)的np数组
        """
        coil_center = self.coil_locator.center_by_net(frame)
        init_ellipse = (coil_center[0], coil_center[1], 100, 100, 0)  # 椭圆初始信息仅包含中心坐标，长短轴和角度忽略(随便填数)
        self.coil_tracer.init_one_cycle(init_ellipse=init_ellipse)
        return coil_center

    def uploading(self, frame):
        """
        在上卷过程中，跟踪检测带卷中心椭圆
        :param frame: 由摄像头获得的带卷RGB图像
        :return: 带卷中心椭圆(x, y, long, short, angle)
        """
        ellipse = self.coil_tracer.analyze_one_frame(frame)  # obtain the center
        return ellipse

    def is_coil_on_car(self, frame):
        """
        检查带卷是否在小车上
        """
        status = self.coil_exist_status.analyze_one_frame(frame)
        return status

    def is_coil_into_unpacker(self, frame):
        """
        检查带卷是否已经插入开卷机就位
        :param frame: 由摄像头获得的带卷RGB图像
        :return: True: 已插入开卷机；False: 未插入开卷机
        """
        status = self.coil_pos_status.analyze_one_frame(frame)
        return status

    def is_coil_unpacked(self, frame):
        """
        检查带卷是否已经开卷
        :param frame: 由摄像头获得的带卷RGB图像
        :return: True: 已开卷；False: 未开卷
        """
        status = self.coil_open_status.analyze_one_frame(frame)
        return status

    def init_one_test_cycle(self, file_name):
        self.init_one_cycle()

        # 打开视频文件
        # 如果需要保存轨迹，则建立新的csv文件
        video = VideoPlayer(file_name)
        if self.save_trace:
            csv_file = open(video.file_name + "_new.csv", "w", newline='')
            self.csv = csv.writer(csv_file)

        # 新视频打开后，默认不保存跟踪视频，除非按下'S'键
        self.saving_video = False
        self.paused = False
        return video

    def control(self):
        self.init_one_cycle()
        start_time = time.perf_counter()
        while True:
            # 从摄像头或文件中读取帧。原视图像为灰度图，为了显示清晰，将用于显示的图像调整为彩色图
            frame_4_proc = self.camera.get_image()
            frame = color_image(frame_4_proc)

            if self.loading_stage != LoadStage.WAITING_4_LOADING:
                self.record_raw_video(frame)

            # 更新PLC状态
            self.plc.refresh_status()

            # 对一帧图像进行控制
            self.control_one_frame(frame=frame_4_proc)

            self.proc_time = (time.perf_counter() - start_time) * 1000
            start_time = time.perf_counter()
            frame = self.show_loading_info(frame)

            # 显示操作按键
            self.show_text(frame, "R(r): save current picture", (20, 250))
            self.show_text(frame, "Q(q): quit", (20, 250 + 60))

            # 视频显示
            frame_resized = cv2.resize(frame, (SCREEN_WIDTH, int(SCREEN_WIDTH * frame.shape[0] / frame.shape[1])))
            cv2.imshow("video", frame_resized)

            if self.loading_stage != LoadStage.WAITING_4_LOADING:
                self.record_proc_video(frame)

            if self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
                self.save_ellipse(self.coil_tracer.coil_ellipse)

            # key operation
            key = cv2.waitKeyEx(0 if self.paused else 10)
            if key == ord('R') or key == ord('r'):
                cv2.imwrite(self.datetime_string() + "-processed.jpg", frame)
                cv2.imwrite(self.datetime_string() + "-original.jpg", frame_4_proc)
            elif key == ord('q') or key == ord('Q'):
                break

    def control_one_frame(self, frame):
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            if self.is_coil_on_car(frame):
                self.wait_for_loading(frame=frame)
                if self.plc.car_up:
                    self.loading_stage = LoadStage.MOVING_TO_UNPACKER
                    self.init_record(frame)
            else:
                self.init_one_cycle()
        elif self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            ellipse = self.uploading(frame=frame)
            if ellipse[0] < AIMING_END:
                self.loading_stage = LoadStage.INTO_UNPACKER
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            status = self.is_coil_into_unpacker(frame=frame)
            if status:
                self.loading_stage = LoadStage.UNPACKING
        elif self.loading_stage == LoadStage.UNPACKING:
            status = self.is_coil_unpacked(frame=frame)
            if status:
                self.init_one_cycle()

    def control_demo(self, frame):
        start_time = time.perf_counter()
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            self.wait_for_loading(frame=frame)
            if self.coil_locator.proc_times > 10:
                self.loading_stage = LoadStage.MOVING_TO_UNPACKER
        elif self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            ellipse = self.uploading(frame=frame)
            if ellipse[0] < AIMING_END:
                self.loading_stage = LoadStage.INTO_UNPACKER
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            status = self.is_coil_into_unpacker(frame=frame)
            if status:
                self.loading_stage = LoadStage.UNPACKING
        elif self.loading_stage == LoadStage.UNPACKING:
            status = self.is_coil_unpacked(frame=frame)
            if status:
                self.loading_stage = LoadStage.UPLOAD_COMPLETED
        self.proc_time = (time.perf_counter() - start_time) * 1000

    def draw_lines(self, frame):
        cv2.line(frame, PositionL1P1, PositionL1P2, (0, 255, 255), 8)
        cv2.line(frame, PositionL2P1, PositionL2P2, (0, 255, 255), 8)
        cv2.line(frame, PositionL3P1, PositionL3P2, (0, 255, 255), 8)

        # 画出三条轨迹线
        cv2.line(frame, start_keypoint, drop_keypoint, (0, 255, 255), 2)
        cv2.line(frame, start_keypoint1, drop_keypoint1, (0, 255, 255), 2)
        cv2.line(frame, drop_keypoint, alignment_keypoint, (0, 255, 255), 2)
        cv2.line(frame, alignment_keypoint, end_keypoint, (0, 255, 255), 2)

        # 画出平移阶段区间线
        cv2.line(frame, stage1_up1, stage1_up2, (20, 20, 255), 2)
        cv2.line(frame, stage1_down1, stage1_down2, (20, 20, 255), 2)
        # 画出下降阶段区间线
        cv2.line(frame, stage2_left1, stage2_left2, (20, 20, 255), 2)
        cv2.line(frame, stage2_right1, stage2_right2, (20, 20, 255), 2)
        cv2.line(frame, stage3_up1, stage3_up2, (20, 20, 255), 2)
        cv2.line(frame, stage3_down1, stage3_down2, (20, 20, 255), 2)

    def show_loading_info(self, frame):
        text_top = 40

        self.draw_lines(frame)

        # 显示时间
        self.show_text(frame, datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S"), (SCREEN_WIDTH - 100, text_top))

        # 显示PLC状态信息
        if not self.demo_ctrl:
            plc_str = "car moving up: %d, down: %d, forward: %d, backward: %d. support opened: %d" % \
                      (self.plc.car_up, self.plc.car_down, self.plc.car_forward, self.plc.car_backward,
                       self.plc.support_open)
            self.show_text(frame, plc_str, (20, text_top))
        text_top += 40

        status_str = ""
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            # 在图像上显示带卷中心位置
            status_str = "Waitting to be loaded, "
            if self.coil_exist_status.status:
                init_center = self.coil_locator.center.astype(np.int32)
                cv2.circle(frame, center=(init_center[0], init_center[1]), radius=5, color=(0, 0, 255), thickness=2)
                status_str += "coil on the car"
            else:
                status_str += "coil not on the car"
        elif self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            status_str = "Moving to the unpacker"

            self.draw_aiming_lines(frame)
            self.coil_tracer.show_trace(frame)

            ellipse = self.coil_tracer.coil_ellipse
            if ellipse is not None:
                status_str += ", coil center = (%d, %d), err(fit/exp) = %.2f, err(cnn/exp) = %.2f, err(cnn/fit) = %.2f" % \
                             (ellipse[0], ellipse[1],
                              self.coil_tracer.err_fit_vs_exp * 100.0,
                              self.coil_tracer.err_cnn_vs_exp * 100.0,
                              self.coil_tracer.err_cnn_vs_fit * 100.0)

                if ellipse[0] < AIMING_BEGIN[0] and ellipse[1] > AIMING_BEGIN[1]:
                    aiming_status = ("position ok", "position low", "position high")
                    status_str += ", " + aiming_status[self.check_aiming_status(ellipse)]
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            status = self.coil_pos_status.status
            status_str = "Loading to the unpacker, " + ("loaded" if status else "not loaded")
        elif self.loading_stage == LoadStage.UNPACKING:
            status = self.coil_open_status.status
            status_str = "Openning the coil, " + "opened" if status else "not opened"
        self.show_text(frame, status_str, (20, text_top))
        text_top += 40

        # 显示计算耗时
        self.show_text(frame, "computing time = %6.2f ms" % self.proc_time, (20, text_top))

        return frame

    def test_demo(self, video_file_name):
        video = self.init_one_test_cycle(file_name=video_file_name)
        while video is not None and not video.is_end():
            # 从摄像头或文件中读取帧，调整尺寸并灰度化
            frame = video.get_frame()
            frame_4_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.control_demo(frame=frame_4_proc)
            frame = self.show_loading_info(frame)

            # 显示当前帧
            self.show_text(frame, "frame count = %d, current frame = %d" % (video.frame_cnt, video.cur_frame), (20, 140))

            # 显示操作按键
            self.show_key_operations(frame, row_begin=200)

            # 视频显示
            frame_resized = cv2.resize(frame, (SCREEN_WIDTH, int(SCREEN_WIDTH * frame.shape[0] / frame.shape[1])))
            cv2.imshow("video", frame_resized)

            # 保存视频
            self.save_video(frame_resized)

            if self.loading_stage == LoadStage.MOVING_TO_UNPACKER and self.save_trace and \
                    self.coil_tracer.coil_ellipse[0] < 1800:
                self.save_ellipse(self.coil_tracer.coil_ellipse)

            # key operation
            key = cv2.waitKeyEx(0 if self.paused else 10)
            if key == ord('p') or key == ord('P'):
                self.paused = not self.paused
            elif key == 2555904:
                if self.paused:
                    video.forward()
            elif key == 2424832:
                if self.paused:
                    video.backward()
            elif key == ord('S') or key == ord('s'):
                self.saving_video = True
            elif key == ord('E') or key == ord('e'):
                self.saving_video = False
            elif key == ord('R') or key == ord('r'):
                cv2.imwrite(self.datetime_string() + ".jpg", frame)
            elif key == ord('L') or key == ord('l'):
                self.loading_stage = 1
            elif key == ord('o') or key == ord('O'):
                self.loading_stage = 2
            elif key == ord('q') or key == ord('Q'):
                break

            if not self.paused:
                video.forward()

    def change_mode(self, mode):
        """
        与张方远的程序兼容
        :param mode:
        :return:
        """
        self.loading_stage = mode

    def run_one_img(self, frame):
        """
        与张方远的程序兼容
        :param frame:
        :return:
        """
        # 椭圆检测功能
        if self.loading_stage == 0 or self.loading_stage == 4:
            # 张方远程序中，带卷的跟踪分为了从堆料区到基坑底部，和向开卷机运动两部分(分别是阶段0和4)。算法改进后不再需要分成两部分，
            # 张方远程序中，包含了对内圆和外圆的检测。外圆检测没有实际用处，因此用内圆数据填充占位。
            ellipse = self.uploading(frame=frame)
            data = [ellipse[0], ellipse[1], ellipse[2], ellipse[3], ellipse[4],
                    ellipse[0], ellipse[1], ellipse[2], ellipse[3], ellipse[4]]
            return data
        # 上卷检测模式
        elif self.loading_stage == 1:
            status = self.coil_pos_status.analyze_one_frame(frame)
            return status
        # 开卷检测模式
        elif self.loading_stage == 2:
            status = self.coil_open_status.analyze_one_frame(frame)
            return status
        # 鞍座检测模式
        elif self.loading_stage == 3:
            status = self.coil_exist_status.analyze_one_frame(frame)
            return status


class combine(object):
    """
    combine类是张方远的程序中设计的，用于被VC程序调用。
    对他的原始程序进行修改，保留接口形式。
    """
    def __init__(self):
        self.load = AutoLoader()
        self.load.init_one_cycle()

    def reset(self):
        self.load.init_one_cycle()

    def receive(self, image, mode):
        """
        接收图片并调用
        :param image:
        :param mode:
        :return:
        """
        self.load.change_mode(mode)
        pp = self.load.run_one_img(image)
        return pp


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files)


def demo_test():
    auto_loader = AutoLoader()
    auto_loader.coil_tracer.read_trace_stat("trace_stat.csv")
    init_pos = None

    print("1: Test on normal cases")
    print("2: Test on bad cases")
    print("3: Analyze inner ellipse parameters")
    print("4: Analyze trace of aiming procedure(high)")
    print("5: Analyze trace of aiming procedure(low)")
    print("6: Analyze trace of aiming procedure(ok)")
    print("7: Test all videos")
    mode = input("")
    if mode == "1":
        # os.chdir("G:\\01 自动上卷视频\\内椭圆参数统计视频")
        os.chdir("DemoVideo")
    elif mode == "2":
        os.chdir("G:\\01 自动上卷视频\\bad cases")
    elif mode == "3":
        auto_loader.save_trace = True
        auto_loader.coil_tracer.exp_ellipse = None
        os.chdir("G:\\01 自动上卷视频\\内椭圆参数统计视频")
    elif mode == "4":
        init_pos = (1405, 1525)
        auto_loader.save_trace = True
        os.chdir("G:\\01 自动上卷视频\\对中视频\\偏高")
    elif mode == "5":
        init_pos = (1405, 1525)
        auto_loader.save_trace = True
        os.chdir("G:\\01 自动上卷视频\\对中视频\\偏低")
    elif mode == "6":
        init_pos = (1405, 1525)
        auto_loader.save_trace = True
        os.chdir("G:\\01 自动上卷视频\\对中视频\\正常")

    if mode != "7":
        for file_name in glob.glob("*.mp4"):
            auto_loader.test_demo(file_name)
    else:
        # 再给定目录下枚举所有的带卷视频，记录带卷轨迹
        auto_loader.save_trace = True
        auto_loader.coil_tracer.exp_ellipse = None
        for root, dirs, files in os.walk("E:\\20220521-20220621数据"):
            for _, file in enumerate(files):
                if os.path.splitext(file)[-1] == ".avi" and "orginal" in os.path.splitext(file)[0]:
                    # 目前保存的视频中，文件名中包含"orginal"的是原始视频
                    # 部分原始视频之前已经分析保存过带卷路径，这些视频将被跳过，不再重复分析
                    if file + ".csv" in files:
                        continue
                    auto_loader.test_demo(video_file_name=root + "\\" + file)


def VC_test():
    cob = combine()


def real_control():
    pass




if __name__ == '__main__':
    # demo_test()

    auto_loader = AutoLoader(demo_ctrl=False)
    auto_loader.coil_tracer.read_trace_stat("trace_stat.csv")
    auto_loader.control()
