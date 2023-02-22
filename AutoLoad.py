import cv2
import time
import os
import glob
import datetime
import csv
import numpy as np
from enum import Enum
import threading

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


class LoadStage(Enum):
    WAITING_4_LOADING  = 0,   # 在堆料区等待上料
    MOVING_TO_UNPACKER = 1,   # 向基坑中移动
    INTO_UNPACKER      = 2,   # 进入开卷机(判断是否到达开卷位置)
    UNPACKING          = 3,   # 开卷(判断是否开卷)
    UPLOAD_COMPLETED   = 4    # 开卷完成


class CameraThread(threading.Thread):
    def __init__(self, demo_video_file=None):
        threading.Thread.__init__(self)
        if demo_video_file is None:
            self.camera = HKCamera()
        else:
            self.camera = None
            self.demo_video = VideoPlayer(demo_video_file)

        self.__lock = threading.RLock()
        self.__event = threading.Event()
        self.__terminated = False
        self.__frame = None
        self.__refreshed = False

        self.paused = False        # 暂停
        self.forward = False       # 视频前放
        self.backward = False      # 视频倒放
        self.frame_time = 0        # 获取一帧图像所花费的时间

    def __read_one_frame(self):
        frame = None
        if self.camera is None:
            if self.demo_video is not None and not self.demo_video.is_end():
                if self.paused:
                    if self.forward:
                        self.demo_video.forward()
                        self.forward = False
                    elif self.backward:
                        self.demo_video.backward()
                        self.backward = False
                else:
                    self.demo_video.forward()
                frame = self.demo_video.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 从视频中读到的是BGR图像，需要转换为灰度图
        else:
            self.camera.refresh()
            frame = self.camera.get_image()
        return frame

    def run(self):
        self.__event.set()
        while not self.__terminated:
            if self.__event.wait(timeout=0.1):
                self.__event.clear()

                begin_time = time.perf_counter()
                frame = self.__read_one_frame()
                self.frame_time = int((time.perf_counter() - begin_time) * 1000)

                self.__lock.acquire()
                self.__frame = frame
                self.__refreshed = True
                self.__lock.release()

    def get_one_frame(self):
        self.__lock.acquire()
        frame = self.__frame.copy() if self.__frame is not None else None
        refreshed = self.__refreshed
        self.__refreshed = False
        self.__lock.release()

        self.__event.set()
        return frame, refreshed

    def close(self):
        self.__terminated = True
        self.join()


class ShowThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.__event = threading.Event()
        self.__terminated = False
        self.__frame_orig = None
        self.__frame_proc = None
        self.__video_orig = None
        self.__video_proc = None
        self.save_record = False
        self.record_root_path = None

    def __init_record(self, frame):
        self.record_root_path = RECORD_ROOT_PATH + "\\" + datetime.datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(self.record_root_path):
            os.makedirs(self.record_root_path)

        time_string = datetime.datetime.now().strftime("%H-%M-%S")
        self.record_root_path += "\\" + time_string
        os.makedirs(self.record_root_path)

        size = (frame.shape[1], frame.shape[0])
        self.__video_orig = cv2.VideoWriter(self.record_root_path + "\\" + time_string + "-original.mp4",
                                            cv2.VideoWriter_fourcc(*'MP4V'), 5, size)
        self.__video_proc = cv2.VideoWriter(self.record_root_path + "\\" + time_string + "-processed.mp4",
                                            cv2.VideoWriter_fourcc(*'MP4V'), 5, size)
        # self.plc.new_csv_record(self.record_root_path + "\\" + time_string + "-csvdata.csv")

    def run(self):
        self.__event.clear()
        while not self.__terminated:
            if not self.__event.wait(timeout=0.1):
                continue
            self.__event.clear()

            if self.save_record:
                if self.__video_orig is None:
                    self.__init_record(self.__frame_proc)
                self.__video_orig.write(color_image(self.__frame_orig))
                self.__video_proc.write(self.__frame_proc)
            else:
                self.__video_orig = None
                self.__video_proc = None

            # 视频显示
            frame_resized = cv2.resize(self.__frame_proc,
                                       (SCREEN_WIDTH, SCREEN_WIDTH * self.__frame_proc.shape[0] // self.__frame_proc.shape[1]))
            cv2.imshow("control monitor", frame_resized)

            # key operation
            cv2.waitKeyEx(10)
            # key = cv2.waitKeyEx(0 if self.camera.paused else 1)
            # if key == ord('R') or key == ord('r'):
            #     cv2.imwrite(self.datetime_string() + "-processed.jpg", frame_bgr)
            #     cv2.imwrite("E:\\ScreenShot\\" + self.datetime_string() + "-processed.jpg", frame_bgr)
            # elif key == ord('q') or key == ord('Q'):
            #     break
            # else:
            #     self.key_operate_demo(key) if RUN_AS_DEMO else self.key_operate_ctrl(key)

    def show_frame(self, frame_orig, frame_proc):
        self.__frame_orig = frame_orig.copy()
        self.__frame_proc = frame_proc.copy()
        self.__event.set()

    def close(self):
        self.__terminated = True
        self.join()


class AutoLoadThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.__terminated = False
        self.__frame_gray = None
        self.__event = threading.Event()

        # 加载各个检测模型
        self.__load_models()
        self.coil_tracker.read_trace_stat("trace_stat.csv")

        self.loading_stage = LoadStage.WAITING_4_LOADING  # 自动上卷的各个阶段
        self.uploading_start_time = None                  # 上卷开始时间
        self.proc_start_time = 0.0                        # 本次控制周期开始时间

        # 各个阶段耗时
        self.read_write_frame_time = 0
        self.track_ctrl_time = 0
        self.show_video_time = 0

    def close(self):
        self.__terminated = True
        self.join()

    def __load_models(self):
        # self.coil_tracker是核心跟踪器
        self.coil_tracker = CenterTracer()
        self.coil_tracker.load_model(PYTHON_PATH + "带卷定位_有干扰_2022-08-05-16-05.pt")
        self.coil_tracker.coil_exist.load_model(PYTHON_PATH + "带卷存在2022-05-25-16-16.pt")
        self.coil_tracker.coil_locator.load_model(PYTHON_PATH + "带卷初始位置2023-02-10-19-37.pt")    # 鞍座检测器，判断带卷是否在鞍座上

        # 上卷检测器
        self.coil_pos_status = CoilPosStatus(src_size=1200, src_position=(0, 800))
        self.coil_pos_status.init_model()
        self.coil_pos_status.load_model(PYTHON_PATH + "上卷检测2020-10-24-16-30.pt")

        # 开卷检测器
        self.coil_open_status = CoilOpenStatus(src_size=1200, src_position=(0, 800))
        self.coil_open_status.init_model()
        self.coil_open_status.load_model(PYTHON_PATH + "开卷检测2020-10-24-12-12.pt")

    def __track_one_frame(self, frame, plc):
        # 在监视模式下，上卷终止有可能判断错误，导致始终处于上卷状态
        # 因此，判断上卷开始后，开始计时，如果3分钟内还没有结束，则强制进入等待上卷状态
        if self.uploading_start_time is not None and (datetime.datetime.now() - self.uploading_start_time).seconds > 600:
            self.loading_stage = LoadStage.WAITING_4_LOADING
            self.coil_tracker.restart_tracking()

        # 目前带卷跟踪仅用于监视带卷位置
        # 带卷跟踪有4个阶段：
        # 1. 等待上卷：带卷位于堆料区(也可能没有带卷)，等待上卷。首先检测判断堆料区是否有带卷(self.coil_tracker.coil_exist)，
        #    之后检测带卷的初始位置(self.coil_tracker.coil_locator)。
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            # 检测跟踪带卷
            # 如果带卷已经跟踪上，但是位置超出正常路径范围，则要求coil_tracker重新定位跟踪带卷
            # 如果带卷到达上卷初始位置，且开卷机打开，则进入上卷状态
            coil, status = self.coil_tracker.analyze_one_frame(frame, plc)
            if status == TrackingStatus.TRACKING and coil is not None:
                if not self.coil_tracker.coil_within_route:
                    self.coil_tracker.restart_tracking()
                # 当带卷位置小于1800时，认为开始上卷
                elif coil[0] < TRACKING_BEGIN_X and (RUN_AS_DEMO or plc.support_open):
                    self.loading_stage = LoadStage.MOVING_TO_UNPACKER
                    self.uploading_start_time = datetime.datetime.now()
        elif self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            # 跟踪带卷位置
            # 如果跟踪中断(回到无带卷，或初始化带卷位置)，则重新回到等待上卷阶段
            # 如果带卷位置小于上卷终止位置，则进入下一阶段，检测带卷是否已装入开卷机
            coil, status = self.coil_tracker.analyze_one_frame(frame, plc)
            if status != TrackingStatus.TRACKING:
                self.loading_stage = LoadStage.WAITING_4_LOADING
                # 注意，这里要告警！！！！
            elif coil[0] < end_keypoint[0]:
                self.loading_stage = LoadStage.INTO_UNPACKER
                self.coil_tracker.restart_tracking()
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            if self.coil_pos_status.analyze_one_frame(frame):
                self.loading_stage = LoadStage.UNPACKING
        elif self.loading_stage == LoadStage.UNPACKING:
            if self.coil_open_status.analyze_one_frame(frame):
                self.loading_stage = LoadStage.WAITING_4_LOADING

    def run(self):
        while not self.__terminated:
            # 基于当前帧图像和PLC反馈信息，进行带卷跟踪
            begin_time = time.perf_counter()
            self.__track_one_frame(frame=self.__frame_gray, plc=self.plc)
            self.car_ctrl()

            # 向PLC写带卷位置q
            if self.coil_tracker.coil_ellipse is not None:
                self.plc.copy_coil_ellipse(self.coil_tracker.coil_ellipse)
                # 没有跟踪数据时上传什么??
            self.track_ctrl_time = int((time.perf_counter() - begin_time) * 1000)

    def analyze_one_frame(self, frame):
        


class AutoLoader(object):
    def __init__(self, run_as_monitor=False):
        self.run_as_monitor = run_as_monitor

        # 初始化plc和摄像头硬件
        self.plc = None
        self.camera = None
        self.__init_hardwares()

        # 建立跟踪线程
        self.__loader = AutoLoadThread()

        # 试验测试中，如果需要存储每个视频的带卷轨迹数据，则设置save_trace=True
        # 正常现场测试时，默认存储带卷轨迹
        self.csv = None
        self.save_trace = False

        # 保存带卷轨迹跟踪视频
        self.default_saving_path = "e:\\NewTest\\"
        self.saving_video = False
        self.video_save = None

        self.video_raw = None      # 原始未处理图像视频
        self.video_proc = None     # 处理后的图像视频

        # 各个阶段耗时
        self.read_write_frame_time = 0
        self.track_ctrl_time = 0
        self.show_video_time = 0

        self.show = ShowThread()
        self.show.start()

    def __del__(self):
        if self.plc is not None:
            self.plc.close()
        if self.camera is not None:
            self.camera.close()
        if self.show is not None:
            self.camera.close()

    def __init_hardwares(self):
        # 初始化并启动plc
        self.plc = PLCComm(RUN_AS_DEMO)
        self.plc.start()

        # 初始化摄像头
        # 在demo模式下，摄像头在每次开始新一个带卷视频时创建
        if not RUN_AS_DEMO:
            try:
                self.camera = CameraThread()
                self.camera.start()
            except Exception as e:
                print(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S:  ") + str(e))
                exit(0)

    @staticmethod
    def datetime_string():
        now = datetime.datetime.now()
        datetime_str = "%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        return datetime_str

    @staticmethod
    def show_text(image, text, pos, color=(0, 255, 0)):
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    def __init_record(self, frame):
        """
        初始化测试记录，包括原始视频、跟踪处理视频，以及csv数据
        :param frame:
        :return:
        """
        self.record_root_path = RECORD_ROOT_PATH + "\\" + datetime.datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(self.record_root_path):
            os.makedirs(self.record_root_path)

        time_string = datetime.datetime.now().strftime("%H-%M-%S")
        self.record_root_path += "\\" + time_string
        os.makedirs(self.record_root_path)

        size = (frame.shape[1], frame.shape[0])
        self.__video_orig = cv2.VideoWriter(self.record_root_path + "\\" + time_string + "-original.mp4",
                                            cv2.VideoWriter_fourcc(*'MP4V'), 5, size)
        self.__video_proc = cv2.VideoWriter(self.record_root_path + "\\" + time_string + "-processed.mp4",
                                            cv2.VideoWriter_fourcc(*'MP4V'), 5, size)
        # self.plc.new_csv_record(self.record_root_path + "\\" + time_string + "-csvdata.csv")

    def control(self):
        self.init_one_cycle()
        self.coil_tracker.restart_tracking()
        self.proc_start_time = time.perf_counter()
        self.snap_shot = True

        while True:
            # 实际控制时，从摄像头中读取一帧。示例和测试时，从视频文件中读取一帧。
            # 摄像头读取的数据为灰度图，为了显示清晰，将用于显示的图像调整为彩色图
            begin_time = time.perf_counter()
            frame_gray, new_frame = self.camera.get_one_frame()
            if not new_frame:
                time.sleep(0.01)
                continue
            if frame_gray is None:
                break
            frame_proc = color_image(frame_gray)

            # 在现场实测时，一旦开始上卷，则记录原始视频
            # if not RUN_AS_DEMO and self.loading_stage != LoadStage.WAITING_4_LOADING and self.video_raw is not None:
            #     self.video_raw.write(frame_bgr)
            self.read_write_frame_time = int((time.perf_counter() - begin_time) * 1000)

            # 基于当前帧图像和PLC反馈信息，进行带卷跟踪
            begin_time = time.perf_counter()
            self.track_one_frame(frame=frame_gray, plc=self.plc)
            self.car_ctrl()

            # 向PLC写带卷位置q
            if self.coil_tracker.coil_ellipse is not None:
                self.plc.copy_coil_ellipse(self.coil_tracker.coil_ellipse)
                # 没有跟踪数据时上传什么??
            self.track_ctrl_time = int((time.perf_counter() - begin_time) * 1000)

            # 显示控制信息和操作按钮
            begin_time = time.perf_counter()
            text_top = self.show_old_loading_info(frame=frame_proc, text_top=60)
            self.show_loading_info(frame=frame_proc, text_top=text_top)
            self.show_key_operations(frame_proc, text_top=text_top + 400)

            # 在现场实测时，一旦开始上卷，同时记录实际控制视频
            if not RUN_AS_DEMO and self.loading_stage != LoadStage.WAITING_4_LOADING:
                # if self.video_proc is not None:
                #     self.video_proc.write(frame_bgr)
                self.plc.add_csv_record()

            # 记录上卷期间带卷跟踪、PLC的状态
            if self.csv is not None and self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
                self.csv.record_tracking(self.coil_tracker)
                self.csv.record_plc(self.plc)
                self.csv.write_file()

            if self.coil_tracker.img_fitting is not None:
                img = self.coil_tracker.img_fitting
                frame_proc[0: img.shape[0], frame_proc.shape[1] - img.shape[1]: frame_proc.shape[1]] = img[:, :]
            if self.coil_tracker.img_exp_fitting is not None:
                img = self.coil_tracker.img_exp_fitting
                frame_proc[img.shape[0]: img.shape[0] * 2, frame_proc.shape[1] - img.shape[1]: frame_proc.shape[1]] \
                    = img[:, :]

            if self.loading_stage != LoadStage.WAITING_4_LOADING:
                self.show.save_record = True
            else:
                self.show.save_record = False
                self.snap_shot = True
            self.show.show_frame(frame_gray, frame_proc)

            # 带卷到达基坑底部后截图
            if SCREEN_SHOT_IN_PIT and not RUN_AS_DEMO and self.coil_tracker.coil_ellipse is not None and \
                    self.coil_tracker.coil_ellipse[1] > 1500 and self.coil_tracker.coil_ellipse[0] < 1420 and \
                    self.snap_shot:
                cv2.imwrite(self.show.record_root_path + "\\" + self.datetime_string() + "-processed.jpg", frame_proc)
                self.snap_shot = False

            # # key operation
            # key = cv2.waitKeyEx(0 if self.camera.paused else 1)
            # if key == ord('R') or key == ord('r'):
            #     cv2.imwrite(self.datetime_string() + "-processed.jpg", frame_bgr)
            #     cv2.imwrite("E:\\ScreenShot\\" + self.datetime_string() + "-processed.jpg", frame_bgr)
            # elif key == ord('q') or key == ord('Q'):
            #     break
            # else:
            #     self.key_operate_demo(key) if RUN_AS_DEMO else self.key_operate_ctrl(key)
            self.show_video_time = int((time.perf_counter() - begin_time) * 1000)













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

    def record_raw_video(self, frame):
        if self.video_raw is None:
            self.init_record(frame)
        self.video_raw.write(frame)

    def record_proc_video(self, frame):
        if self.video_proc is None:
            self.init_record(frame)
        self.video_proc.write(frame)



    def show_key_operations(self, frame, text_top):
        row = text_top
        if RUN_AS_DEMO:
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
            self.show_text(frame, "s(S): manually start", (20, row))
            row += 60
            self.show_text(frame, "E(e): manually stop", (20, row))
            row += 60
            self.show_text(frame, "F(f): show edge", (20, row))
            row += 60
        self.show_text(frame, "Q(q): quit", (20, row))
        row += 60
        return row

    def key_operate_ctrl(self, key):
        if key == ord('s') or key == ord('S'):
            self.loading_stage = LoadStage.MOVING_TO_UNPACKER
        elif key == ord('e') or key == ord('E'):
            self.loading_stage = LoadStage.WAITING_4_LOADING
        elif key in (ord('f'), ord('F')):
            self.coil_tracker.show_fitting = not self.coil_tracker.show_fitting

    def key_operate_demo(self, key):
        if key == ord('p') or key == ord('P'):
            self.camera.paused = not self.camera.paused
        elif key == 2555904:
            if self.camera.paused:
                self.camera.forward = True
        elif key == 2424832:
            if self.camera.paused:
                self.camera.backward = True
        elif key == ord('S') or key == ord('s'):
            self.saving_video = True
        elif key == ord('E') or key == ord('e'):
            self.saving_video = False
        elif key == ord('L') or key == ord('l'):
            self.loading_stage = 1
        elif key == ord('o') or key == ord('O'):
            self.loading_stage = 2

    def init_one_cycle(self):
        """
        准备开始新带卷的自动上卷控制。只需要上卷时调用一次，也可多次调用，对上卷无影响。
        :return: 无
        """
        self.loading_stage = LoadStage.WAITING_4_LOADING
        self.uploading_start_time = None
        self.video_raw = None
        self.video_proc = None
        self.csv = None

    def track_one_frame(self, frame, plc):
        # 在监视模式下，上卷终止有可能判断错误，导致始终处于上卷状态
        # 因此，判断上卷开始后，开始计时，如果3分钟内还没有结束，则强制进入等待上卷状态
        if self.uploading_start_time is not None and (datetime.datetime.now() - self.uploading_start_time).seconds > 600:
            self.loading_stage = LoadStage.WAITING_4_LOADING
            self.coil_tracker.restart_tracking()

        # 目前带卷跟踪仅用于监视带卷位置
        # 带卷跟踪有4个阶段：
        # 1. 等待上卷：带卷位于堆料区(也可能没有带卷)，等待上卷。首先检测判断堆料区是否有带卷(self.coil_tracker.coil_exist)，
        #    之后检测带卷的初始位置(self.coil_tracker.coil_locator)。
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            self.init_one_cycle()

            # 检测跟踪带卷，如果带卷不存在，则返回None
            coil, status = self.coil_tracker.analyze_one_frame(frame, plc)
            if status == TrackingStatus.TRACKING and coil is not None:
                if not self.coil_tracker.coil_within_route:
                    self.coil_tracker.restart_tracking()
                # 当带卷位置小于1800时，认为开始上卷
                elif coil[0] < 1800 and (RUN_AS_DEMO or plc.support_open):
                    self.loading_stage = LoadStage.MOVING_TO_UNPACKER
                    self.uploading_start_time = datetime.datetime.now()
        elif self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            # 跟踪带卷位置
            # 如果跟踪中断(回到无带卷，或初始化带卷位置)，则重新回到等待上卷阶段
            # 如果带卷位置小于上卷终止位置，则进入下一阶段，检测带卷是否已装入开卷机
            coil, status = self.coil_tracker.analyze_one_frame(frame, plc)
            if status != TrackingStatus.TRACKING:
                self.loading_stage = LoadStage.WAITING_4_LOADING
            elif coil is not None and coil[0] < end_keypoint[0]:
                self.loading_stage = LoadStage.INTO_UNPACKER
                self.coil_tracker.restart_tracking()
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            if self.coil_pos_status.analyze_one_frame(frame):
                self.loading_stage = LoadStage.UNPACKING
        elif self.loading_stage == LoadStage.UNPACKING:
            if self.coil_open_status.analyze_one_frame(frame):
                self.loading_stage = LoadStage.WAITING_4_LOADING

    def car_ctrl(self):
        pass

    def write_plc(self):
        pass

    def draw_lines(self, frame):
        cv2.line(frame, PitMarkL1P1, PitMarkL1P2, (0, 255, 255), 8)
        cv2.line(frame, PitMarkL2P1, PitMarkL2P2, (0, 255, 255), 8)
        cv2.line(frame, PitMarkL3P1, PitMarkL3P2, (0, 255, 255), 8)

        # 画出三条轨迹线
        cv2.line(frame, coil_trans_begin, coil_fall_begin, (0, 255, 255), 2)
        # cv2.line(frame, coil_trans_begin1, coil_fall_begin1, (0, 255, 255), 2)
        cv2.line(frame, coil_fall_begin, alignment_keypoint, (0, 255, 255), 2)
        cv2.line(frame, alignment_keypoint, end_keypoint, (0, 255, 255), 2)

        # 画出平移阶段区间线
        cv2.line(frame, stage1_up1, stage1_up2, (20, 20, 255), 2)
        cv2.line(frame, stage1_down1, stage1_down2, (20, 20, 255), 2)

        # 画出下降阶段区间线
        cv2.line(frame, stage2_left1, stage2_left2, (20, 20, 255), 2)
        cv2.line(frame, stage2_right1, stage2_right2, (20, 20, 255), 2)

        cv2.line(frame, stage3_up1, stage3_up2, (20, 20, 255), 2)
        cv2.line(frame, stage3_down1, stage3_down2, (20, 20, 255), 2)

    def show_loading_info(self, frame, text_top):
        # 显示当前时间
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d, %H:%M:%S:") + ("%03d" % int(now.microsecond * 0.001))
        self.show_text(frame, time_str, (frame.shape[1] - 600, 40))

        # 显示检测跟踪算法的主要耗时
        proc_time = int((time.perf_counter() - self.proc_start_time) * 1000)
        self.proc_start_time = time.perf_counter()

        self.show_text(frame, "Time consumption", (20, text_top))
        self.show_text(frame, "Total   %4d     Camera: %4d" % (proc_time, self.read_write_frame_time),
                       (20, text_top + 40))
        self.show_text(frame, "Edge:   %4d     CNN:     %4d" % (self.coil_tracker.edge_time, self.coil_tracker.cnn_time),
                       (20, text_top + 80))
        self.show_text(frame, "Fit:    %4d     Fit exp: %4d    Offset: %4d" %
                       (self.coil_tracker.fit_time, self.coil_tracker.fit_exp_time, self.coil_tracker.offset_time),
                       (20, text_top + 120))
        self.show_text(frame, "Track:   %4d    Show:    %4d" % (self.track_ctrl_time, self.show_video_time),
                       (20, text_top + 160))
        text_top += 200

        # 显示相机状态信息
        # if not RUN_AS_DEMO and self.loading_stage != LoadStage.MOVING_TO_UNPACKER:
        #     exp_time = self.camera.get_exposure_time()
        #     self.show_text(frame, "Camera exposure time = %1.1fms" % (exp_time * 0.001), (20, text_top))
        #     text_top += 40

        # demo状态下，显示视频总帧数和当前帧
        if RUN_AS_DEMO:
            self.show_text(frame, "Video frame count = %d, current frame = %d" %
                           (self.camera.demo_video.frame_cnt, self.camera.demo_video.cur_frame), (20, text_top))
            text_top += 40

        # 显示带卷中心附近图像的直方图
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

        # 绘制定位辅助线
        self.draw_lines(frame)

        # 正在跟踪带卷，显示跟踪状态信息
        status_str = ""
        if self.coil_tracker.tracking_status == TrackingStatus.TRACKING:
            # 带卷跟踪中，显示带卷的中心轨迹，以及跟踪误差
            status_str = "Tracking coil"
            self.coil_tracker.show_trace(frame)
            ellipse = self.coil_tracker.coil_ellipse
            if ellipse is not None:
                status_str += ", center = (%d, %d), ellipse error = %5.1f, movement error = %5.1f" % \
                              (ellipse[0], ellipse[1],
                               self.coil_tracker.err_ellipse * 100.0,
                               self.coil_tracker.err_movement * 100.0)
        elif self.coil_tracker.tracking_status == TrackingStatus.NO_COIL:
            status_str = "No coil"
        elif self.coil_tracker.tracking_status == TrackingStatus.LOCATING:
            status_str = "Initializing coil center"
            init_center = self.coil_tracker.coil_locator.center
            if init_center is not None:
                cv2.circle(frame, center=(init_center[0], init_center[1]), radius=5, color=(0, 0, 255), thickness=2)
        self.show_text(frame, status_str, (20, text_top))
        text_top += 40

        status_str = ""
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            # 在图像上显示带卷中心位置
            status_str = "Waiting to be loaded"
        elif self.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            status_str = "Moving to the unpacker"
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            status_str = "Loading to the unpacker, " + ("loaded" if self.coil_pos_status.status else "not loaded")
        elif self.loading_stage == LoadStage.UNPACKING:
            status_str = "Opening the coil, " + "opened" if self.coil_open_status.status else "not opened"
        self.show_text(frame, status_str, (20, text_top))
        text_top += 40

        return text_top

    def show_old_loading_info(self, frame, text_top):
        self.show_text(frame, "Mode0:", (10, text_top + 40), (20, 255, 255))
        self.show_text(frame, "Rbt :", (10, text_top + 90), (20, 255, 255))
        self.show_text(frame, "%", (240, text_top + 90), (20, 255, 255))

        # 小车行动命令
        self.show_text(frame, "In  : %d" % self.plc.car_forward , (10, text_top + 140), (20, 255, 255))
        self.show_text(frame, "Out : %d" % self.plc.car_backward, (10, text_top + 190), (20, 255, 255))
        self.show_text(frame, "Up  : %d" % self.plc.car_up,       (10, text_top + 240), (20, 255, 255))
        self.show_text(frame, "Down: %d" % self.plc.car_down,     (10, text_top + 290), (20, 255, 255))

        # 卷筒状态: 1：涨径; 0：缩径
        self.show_text(frame, "Expand: %d" % self.plc.PayoffCoilBlock_Expand, (300, text_top + 40), (20, 255, 255))

        # 压辊状态: 1：压下; 0：抬起
        self.show_text(frame, "jamroll: %d" % self.plc.JammingRoll_State, (300, text_top + 90), (20, 255, 255))

        # 支撑状态: 1：打开; 0：关闭
        self.show_text(frame, "support: %d" % self.plc.PayoffSupport_Open, (300, text_top + 140), (20, 255, 255))

        # 导板状态: 1：抬起; 0：放下
        self.show_text(frame, "board  : %d" % self.plc.GuideBoard_Up, (300, text_top + 190), (20, 255, 255))

        # 卷筒旋转: 1：正转; 2：反转; 0：不动
        if self.plc.Payoff_PositiveRun:
            self.show_text(frame, "block  : 1", (300, text_top + 240), (20, 255, 255))
        elif self.plc.Payoff_NegativeRun:
            self.show_text(frame, "block  : 2", (300, text_top + 240), (20, 255, 255))
        else:
            self.show_text(frame, "block  : 0", (300, text_top + 240), (20, 255, 255))

        # 导板伸缩: 1：伸出; 0：缩回
        self.show_text(frame, "tongue : %d" % self.plc.GuideBoard_Extend, (300, text_top + 290), (20, 255, 255))

        # 内径圆圆心坐标
        ellipse = self.coil_tracker.coil_ellipse
        if ellipse is not None:
            self.show_text(frame, "x: %d" % ellipse[0], (10, text_top + 360), (20, 255, 255))
            self.show_text(frame, "y: %d" % ellipse[1], (300, text_top + 360), (20, 255, 255))
            """
            # 模糊度百分比
            self.show_text(frame, to_string(D.definition_rate), (2900, 1970), (20, 255, 255))
            self.show_text(frame, "%", (3000, 1970), (20, 255, 255))
            """

        return text_top + 360 + 40

    def test_demo(self, video_file_name):
        # 打开视频文件
        # 如果需要保存轨迹，则建立新的csv文件
        self.video_name = video_file_name
        self.camera = CameraThread(video_file_name)
        self.camera.start()
        # self.demo_video = VideoPlayer(video_file_name)
        if self.save_trace:
            csv_file = open(self.demo_video.file_name + "_new.csv", "w", newline='')
            self.csv = csv.writer(csv_file)

        # 新视频打开后，默认不保存跟踪视频，除非按下'S'键
        self.saving_video = False

        self.control()
        self.camera.close()


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
    # mode = input("")
    mode = "1"
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
        for file_name in glob.glob("*.avi"):
            auto_loader.test_demo(file_name)
    else:
        # 再给定目录下枚举所有的带卷视频，记录带卷轨迹
        auto_loader.save_trace = True
        #auto_loader.coil_tracker.exp_ellipse = None
        for root, dirs, files in os.walk("E:\\20220521-20220621数据\\视频数据\\2022-06-19"):        #"E:\\20220521-20220621数据"):
            for _, file in enumerate(files):
                if os.path.splitext(file)[-1] in (".avi", ".mp4") and "orginal" in os.path.splitext(file)[0]:
                    # 目前保存的视频中，文件名中包含"orginal"的是原始视频
                    # 部分原始视频之前已经分析保存过带卷路径，这些视频将被跳过，不再重复分析
                    # if file + ".csv" in files:
                    #     continue
                    auto_loader.test_demo(video_file_name=root + "\\" + file)


def real_control():
    auto_loader = AutoLoader()
    auto_loader.coil_tracker.read_trace_stat("trace_stat.csv")
    auto_loader.control()


if __name__ == '__main__':
    if RUN_AS_DEMO:
        demo_test()
    else:
        real_control()
