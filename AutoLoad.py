import cv2
import time
import os
import glob
import datetime
import csv
import numpy as np
from enum import Enum

from CenterTracer import *
from CoilStatus import CoilOpenStatus
from CoilStatus import CoilPosStatus
from PLCComm import PLCComm
from HKCamera import HKCamera
from CoilImageProc import *
from TraceCSV import TraceCSV
from FilePath import PYTHON_PATH
from FixedParams import *


SCREEN_WIDTH = 1600
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
    def __init__(self, run_as_demo=True, run_as_monitor=False):
        self.run_as_demo = run_as_demo
        self.run_as_monitor = run_as_monitor

        # 加载各个检测模型
        self.__load_models()

        # 初始化实际控制使用的硬件
        # 如果硬件打开失败，则退出
        if not run_as_demo and not self.__init_hardwares():
            exit(0)

        self.demo_video = None    # 当前的demo视频

        # 以下变量主要在demo中使用到
        self.loading_stage = LoadStage.WAITING_4_LOADING     # 自动上卷的各个阶段
        self.proc_start_time = 0.0                           # 本次控制周期开始时间

        # 试验测试中，如果需要存储每个视频的带卷轨迹数据，则设置save_trace=True
        # 正常现场测试时，默认存储带卷轨迹
        self.csv = None
        self.save_trace = False

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

        self.loading_trigged = False
        self.equalize_image = False
        self.uploading_start_time = None     # 上卷开始时间

    def __load_models(self):
        # self.coil_tracker是核心跟踪器
        self.coil_tracker = CenterTracer()
        self.coil_tracker.load_model(PYTHON_PATH + "带卷定位_有干扰_2022-08-05-16-05.pt")
        self.coil_tracker.coil_exist.load_model(PYTHON_PATH + "带卷存在2022-05-25-16-16.pt")
        self.coil_tracker.coil_locator.load_model(PYTHON_PATH + "带卷初始位置2023-01-14-11-44.pt")    # 鞍座检测器，判断带卷是否在鞍座上

        # 上卷检测器
        self.coil_pos_status = CoilPosStatus(src_size=1200, src_position=(0, 800))
        self.coil_pos_status.init_model()
        self.coil_pos_status.load_model(PYTHON_PATH + "上卷检测2020-10-24-16-30.pt")

        # 开卷检测器
        self.coil_open_status = CoilOpenStatus(src_size=1200, src_position=(0, 800))
        self.coil_open_status.init_model()
        self.coil_open_status.load_model(PYTHON_PATH + "开卷检测2020-10-24-12-12.pt")

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

        size = (frame.shape[1], frame.shape[0])
        self.video_raw = cv2.VideoWriter(file_path_name + "-original.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 5, size)
        self.video_proc = cv2.VideoWriter(file_path_name + "-processed.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 5, size)

        self.csv = TraceCSV(file_name=file_path_name)

    def record_raw_video(self, frame):
        self.video_raw.write(frame)

    def record_proc_video(self, frame):
        self.video_proc.write(frame)

    def show_key_operations(self, frame, row_begin):
        row = row_begin
        if self.run_as_demo:
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
        else:
            self.show_text(frame, "R(r): save current picture", (20, row))
            row += 60
            self.show_text(frame, "Q(q): quit", (20, row))
            row += 60
            self.show_text(frame, "s(S): manually start", (20, row))
            row += 60
            self.show_text(frame, "t(T): manually stop", (20, row))
            row += 60
            self.show_text(frame, "f(F): show edge", (20, row))
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

    def get_one_frame(self):
        if self.run_as_demo:
            frame = self.demo_video.get_frame()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.camera.refresh()
            return self.camera.get_image()

    def init_one_cycle(self):
        """
        准备开始新带卷的自动上卷控制。只需要上卷时调用一次，也可多次调用，对上卷无影响。
        :return: 无
        """
        # init_ellipse = (INIT_COIL_CENTER[0], INIT_COIL_CENTER[1], 0, 0, 0)
        # self.coil_locator.init_one_cycle()
        # self.coil_tracker.init_one_cycle(init_ellipse=init_ellipse)
        self.loading_stage = LoadStage.WAITING_4_LOADING

    def wait_for_loading(self, frame):
        """
        带卷位于堆料区等待上卷，持续定位其初始坐标(并滤波)，将初始位置写入跟踪器初始位置中
        :param frame: 由摄像头获得的带卷RGB图像
        :return: 带卷的初始位置，是一个(int, int)的np数组
        """
        coil_center = self.coil_locator.center_by_net(frame)
        init_ellipse = (coil_center[0], coil_center[1], 100, 100, 0)  # 椭圆初始信息仅包含中心坐标，长短轴和角度忽略(随便填数)
        self.coil_tracker.init_one_cycle(init_ellipse=init_ellipse)
        return coil_center

    def track_coil(self, frame, plc):
        """
        在上卷过程中，跟踪检测带卷中心椭圆
        :param frame: 由摄像头获得的带卷RGB图像
        :return: 带卷中心椭圆(x, y, long, short, angle)
        """
        self.coil_tracker.analyze_one_frame(frame, plc)  # obtain the center

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
        self.proc_start_time = time.perf_counter()

        while True:
            # 实际控制时，从摄像头中读取一帧。示例和测试时，从视频文件中读取一帧。
            # 摄像头读取的数据为灰度图，为了显示清晰，将用于显示的图像调整为彩色图
            frame_gray = self.get_one_frame()
            frame_bgr = color_image(frame_gray)

            # 在现场实测时，一旦开始上卷，则记录原始视频
            if not self.run_as_demo and self.loading_stage != LoadStage.WAITING_4_LOADING:
                self.record_raw_video(frame_bgr)

            # 更新PLC状态
            if self.plc is not None:
                self.plc.refresh_status()

            # 基于当前帧图像和PLC反馈信息，进行带卷跟踪
            self.track_one_frame(frame=frame_gray, plc=self.plc)
            self.car_ctrl()

            # 显示控制信息和操作按钮
            frame_bgr = self.show_loading_info(frame=frame_bgr, top=40)
            frame_bgr = self.show_key_operations(frame_bgr, row_begin=400)

            # 在现场实测时，一旦开始上卷，同时记录实际控制视频
            if not self.run_as_demo and self.loading_stage != LoadStage.WAITING_4_LOADING:
                self.record_proc_video(frame_bgr)

            # 记录上卷期间带卷跟踪、PLC的状态
            if self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
                self.csv.record_tracking(self.coil_tracker)
                self.csv.record_plc(self.plc)
                self.csv.write_file()

            # 视频显示
            frame_resized = cv2.resize(frame_bgr,
                                       (SCREEN_WIDTH, int(SCREEN_WIDTH * frame_bgr.shape[0] / frame_bgr.shape[1])))
            cv2.imshow("control monitor", frame_resized)

            # key operation
            key = cv2.waitKeyEx(0 if self.paused else 10)
            if key == ord('R') or key == ord('r'):
                cv2.imwrite(self.datetime_string() + "-processed.jpg", frame)
                cv2.imwrite(self.datetime_string() + "-original.jpg", frame_gray)
            elif key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                self.loading_stage = LoadStage.MOVING_TO_UNPACKER
                self.init_record(frame)
            elif key == ord('t') or key == ord('T'):
                self.loading_stage = LoadStage.WAITING_4_LOADING
            elif key == ord('e') or key == ord('E'):
                self.equalize_image = not self.equalize_image
            elif key in (ord('f'), ord('F')):
                self.coil_tracker.show_fitting = not self.coil_tracker.show_fitting

    def track_one_frame(self, frame, plc):
        # 在监视模式下，上卷终止有可能判断错误，导致始终处于上卷状态
        # 因此，判断上卷开始后，开始计时，如果1分钟内还没有结束，则强制进入等待上卷状态
        if self.uploading_start_time is not None and time.perf_counter() - self.uploading_start_time > 60.0:
            self.loading_stage = LoadStage.WAITING_4_LOADING

        # 目前带卷跟踪仅用于监视带卷位置
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            self.uploading_start_time = None

            # 检测跟踪带卷，如果带卷不存在，则返回None
            self.coil_tracker.analyze_one_frame(frame, plc)
            if self.coil_tracker.coil_ellipse is None:
                return

            # 当带卷位置小于1800时，认为开始上卷
            if self.coil_tracker.coil_ellipse[0] < 1800:
                self.loading_stage = LoadStage.MOVING_TO_UNPACKER
                self.uploading_start_time = time.perf_counter()
                if not self.run_as_demo:
                    self.init_record(frame)
        elif self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            # 跟踪带卷位置
            # 如果跟踪中断(回到无带卷，或初始化带卷位置)，则重新回到等待上卷阶段
            self.coil_tracker.analyze_one_frame(frame, plc)
            if self.coil_tracker.tracking_status in (TrackingStatus.no_coil, TrackingStatus.init_position):
                self.loading_stage = LoadStage.WAITING_4_LOADING
                return

            # 如果带卷位置小于上卷终止位置，则检测带卷是否已装入开卷机
            if self.coil_tracker.coil_ellipse[0] < end_keypoint[0]:
                self.loading_stage = LoadStage.INTO_UNPACKER
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            status = self.coil_pos_status.analyze_one_frame(frame)
            if status:
                self.loading_stage = LoadStage.UNPACKING
        elif self.loading_stage == LoadStage.UNPACKING:
            status = self.coil_open_status.analyze_one_frame(frame)
            if status:
                self.loading_stage = LoadStage.WAITING_4_LOADING

    def car_ctrl(self):
        pass

    def control_demo(self, frame):
        start_time = time.perf_counter()
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            self.wait_for_loading(frame=frame)
            if self.coil_locator.proc_times > 10:
                self.loading_stage = LoadStage.MOVING_TO_UNPACKER
        elif self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            ellipse = self.track_coil(frame=frame)
            if ellipse[0] < AIMING_END:
                self.loading_stage = LoadStage.INTO_UNPACKER
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            status = self.coil_pos_status.analyze_one_frame(frame)
            if status:
                self.loading_stage = LoadStage.UNPACKING
        elif self.loading_stage == LoadStage.UNPACKING:
            status = self.coil_open_status.analyze_one_frame(frame)
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

    def show_loading_info(self, frame, top):
        text_top = 40

        self.draw_lines(frame)

        # 显示当前时间
        self.show_text(frame, datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S"), (frame.shape[1] - 600, text_top))

        # 显示PLC状态信息
        if self.plc is not None:
            plc_str = "plc heartbeat: %d, car moving up: %d, down: %d, forward: %d, backward: %d. support opened: %d" % \
                      (self.plc.heartbeat, self.plc.car_up, self.plc.car_down, self.plc.car_forward,
                       self.plc.car_backward, self.plc.support_open)
            self.show_text(frame, plc_str, (20, text_top))
            text_top += 40

        # 显示相机状态信息
        if self.camera is not None:
            if self.loading_stage != LoadStage.MOVING_TO_UNPACKER:
                exp_time = self.camera.get_exposure_time()
                self.show_text(frame, "Camera exposure time = %1.1fms" % (exp_time * 0.001), (20, text_top))
                text_top += 40

        center = None
        if self.coil_tracker.coil_ellipse is not None:
            center = self.coil_tracker.coil_ellipse
        elif self.coil_tracker.coil_locator.center is not None:
            center = self.coil_tracker.coil_locator.center
        if center is not None:
            image = frame[center[1] - 256: center[1] + 256, center[0] - 256: center[0] + 256]
            _, dist = image_histogram(image)
            hist_img = color_image(draw_histogram(dist))
            frame[text_top: text_top + hist_img.shape[0], 20: 20 + hist_img.shape[1]] = hist_img[:, :]
            text_top += 100 + 40

        # 正在跟踪带卷，显示跟踪状态信息
        status_str = ""
        if self.coil_tracker.tracking_status == TrackingStatus.tracking:
            status_str = "Tracking coil"
            self.coil_tracker.show_trace(frame)
            ellipse = self.coil_tracker.coil_ellipse
            if ellipse is not None:
                status_str += ", center = (%d, %d), ellipse error = %5.1f, movement error = %5.1f" % \
                              (ellipse[0], ellipse[1],
                               self.coil_tracker.err_ellipse * 100.0,
                               self.coil_tracker.err_movement * 100.0)
        elif self.coil_tracker.tracking_status == TrackingStatus.no_coil:
            status_str = "No coil"
        elif self.coil_tracker.tracking_status == TrackingStatus.init_position:
            status_str = "Initializing coil center"
            init_center = self.coil_tracker.coil_locator.center
            if init_center is not None:
                cv2.circle(frame, center=(init_center[0], init_center[1]), radius=5, color=(0, 0, 255), thickness=2)
        self.show_text(frame, status_str, (20, text_top))
        text_top += 40

        status_str = ""
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            # 在图像上显示带卷中心位置
            status_str = "Waitting to be loaded"
            # if self.camera is not None:
            #     exp_time = self.camera.get_exposure_time()
            #     status_str += ", camera exposure time = %1.1fms" % (exp_time * 0.001)
            # if self.coil_exist_status.status:
            #     init_center = self.coil_locator.center.astype(np.int32)
            #     cv2.circle(frame, center=(init_center[0], init_center[1]), radius=5, color=(0, 0, 255), thickness=2)
            #     status_str += "coil on the car"
            # else:
            #     status_str += "coil not on the car"
        elif self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            status_str = "Moving to the unpacker"

            # self.coil_tracker.show_trace(frame)
            # ellipse = self.coil_tracker.coil_ellipse
            # if ellipse is not None:
            #     status_str += ", coil center = (%d, %d), ellipse error = %5.1f, movement error = %5.1f" % \
            #                  (ellipse[0], ellipse[1],
            #                   self.coil_tracker.err_ellipse * 100.0,
            #                   self.coil_tracker.err_movement * 100.0)
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            status = self.coil_pos_status.status
            status_str = "Loading to the unpacker, " + ("loaded" if status else "not loaded")
        elif self.loading_stage == LoadStage.UNPACKING:
            status = self.coil_open_status.status
            status_str = "Openning the coil, " + "opened" if status else "not opened"
        self.show_text(frame, status_str, (20, text_top))
        text_top += 40

        # 显示控制周期耗时
        proc_time = (time.perf_counter() - self.proc_start_time) * 1000
        self.proc_start_time = time.perf_counter()
        self.show_text(frame, "Processing time = %6.2f ms" % proc_time, (20, text_top))

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
                    self.coil_tracker.coil_ellipse[0] < 1800:
                self.save_ellipse(self.coil_tracker.coil_ellipse)

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
            ellipse = self.track_coil(frame=frame)
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
    auto_loader.coil_tracker.read_trace_stat("trace_stat.csv")
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
        auto_loader.coil_tracker.exp_ellipse = None
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
        auto_loader.coil_tracker.exp_ellipse = None
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

    auto_loader = AutoLoader(run_as_demo=False)
    auto_loader.coil_tracker.read_trace_stat("trace_stat.csv")
    auto_loader.control()
