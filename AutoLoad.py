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
            # self.camera.set_Value(param_type="float_value", node_name="FrameRate", node_value=6)
            # self.camera.set_Value(param_type="float_value", node_name="ExposureTime", node_value=100000)
            # self.camera.set_exposure_time(100000)
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

                self.__lock.acquire()
                begin_time = time.perf_counter()
                frame = self.__read_one_frame()
                self.frame_time = int((time.perf_counter() - begin_time) * 1000)

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
        return frame, refreshed, self.frame_time

    def close(self):
        self.__terminated = True
        self.join()


class ShowThread(threading.Thread):
    def __init__(self, loader, plc):
        threading.Thread.__init__(self)
        self.__event = threading.Event()
        self.__loader = loader
        self.__plc = plc
        self.__terminated = False
        self.__frame_orig = None
        self.__frame_proc = None
        self.__video_orig = None
        self.__video_proc = None
        self.__lst_stage = LoadStage.WAITING_4_LOADING
        self.__snap_shot = False

        self.__process_time = 0
        self.__frame_time = 0
        self.__read_write_frame_time = 0
        self.__track_ctrl_time = 0
        self.__show_video_time = 0
        self.__edge_time = 0
        self.__cnn_time = 0
        self.__fit_time = 0
        self.__fit_exp_time = 0
        self.__offset_time = 0

        self.save_record = False          # 是否存储视频和csv记录
        self.record_root_path = None      # 当前记录路径
        self.key_input = 0                # 窗口按键

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

        self.__plc.new_csv_record(self.record_root_path + "\\" + time_string + "-csvdata.csv")

    @staticmethod
    def __datetime_string():
        now = datetime.datetime.now()
        datetime_str = "%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        return datetime_str

    @staticmethod
    def __show_text(image, text, pos, color=(0, 255, 0)):
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    @staticmethod
    def __draw_lines(frame):
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

    @staticmethod
    def __show_key_operations(frame, text_top):
        row = text_top
        if RUN_AS_DEMO:
            ShowThread.__show_text(frame, "P(p): pause", (20, row))
            row += 60
            ShowThread.__show_text(frame, "->: move forward", (20, row))
            row += 60
            ShowThread.__show_text(frame, "<-: move backward", (20, row))
            row += 60
            ShowThread.__show_text(frame, "S(s): start tracing and recording", (20, row))
            row += 60
            ShowThread.__show_text(frame, "E(e): stop tracing and recording", (20, row))
            row += 60
            ShowThread.__show_text(frame, "R(r): save current picture", (20, row))
            row += 60
            ShowThread.__show_text(frame, "L(l): detect if the coil is loaded", (20, row))
            row += 60
            ShowThread.__show_text(frame, "O(o): detect if the coil is opened", (20, row))
            row += 60
            ShowThread.__show_text(frame, "g(G): show coil edge", (20, row))
            row += 60
        else:
            ShowThread.__show_text(frame, "R(r): save current picture", (20, row))
            row += 60
            ShowThread.__show_text(frame, "s(S): manually start", (20, row))
            row += 60
            ShowThread.__show_text(frame, "E(e): manually stop", (20, row))
            row += 60
            ShowThread.__show_text(frame, "F(f): show edge", (20, row))
            row += 60
        ShowThread.__show_text(frame, "Q(q): quit", (20, row))
        row += 60
        return row

    def __draw_fitting(self, frame):
        img = self.__loader.coil_tracker.img_fitting
        if img is not None:
            frame[0: img.shape[0], frame.shape[1] - img.shape[1]: frame.shape[1]] = img[:, :]

        img = self.__loader.coil_tracker.img_exp_fitting
        if img is not None:
            frame[img.shape[0]: img.shape[0] * 2, frame.shape[1] - img.shape[1]: frame.shape[1]] = img[:, :]

    def __draw_coil_trace(self, frame):
        if self.__loader.coil_tracker.coil_ellipse is None:
            return

        ellipse = self.__loader.coil_tracker.coil_ellipse
        cv2.ellipse(frame, (ellipse[0], ellipse[1]), (ellipse[2], ellipse[3]), ellipse[4], 0, 360, (0, 255, 0), 2)

        # show center moving trace
        if len(self.__loader.coil_tracker.kalman_trace) > 0:
            lst_ellipse = self.__loader.coil_tracker.kalman_trace[0]
            for _, ellipse in enumerate(self.__loader.coil_tracker.kalman_trace):
                p1 = (lst_ellipse[0], lst_ellipse[1])
                p2 = (ellipse[0], ellipse[1])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                lst_ellipse = ellipse

    def __draw_histgram(self, orig_frame, proc_frame, top):
        # 显示带卷中心附近图像的直方图
        if self.__loader.coil_tracker.coil_ellipse is not None:
            center = self.__loader.coil_tracker.coil_ellipse
            if center is not None:
                image = orig_frame[center[1] - 256: center[1] + 256, center[0] - 256: center[0] + 256]
                _, dist = image_histogram(image)
                hist_img = color_image(draw_histogram(dist))
                proc_frame[top: top + hist_img.shape[0], 20: 20 + hist_img.shape[1]] = hist_img[:, :]
        return top + 100 + 60

    def __show_old_loading_info(self, frame, text_top):
        self.__show_text(frame, "Mode0:", (20, text_top + 40), (20, 255, 255))
        self.__show_text(frame, "Rbt :", (20, text_top + 90), (20, 255, 255))
        self.__show_text(frame, "%", (240, text_top + 90), (20, 255, 255))

        # 小车行动命令
        self.__show_text(frame, "In  : %d" % self.__plc.car_forward , (20, text_top + 140), (20, 255, 255))
        self.__show_text(frame, "Out : %d" % self.__plc.car_backward, (20, text_top + 190), (20, 255, 255))
        self.__show_text(frame, "Up  : %d" % self.__plc.car_up,       (20, text_top + 240), (20, 255, 255))
        self.__show_text(frame, "Down: %d" % self.__plc.car_down,     (20, text_top + 290), (20, 255, 255))

        # 卷筒状态: 1：涨径; 0：缩径
        self.__show_text(frame, "Expand: %d" % self.__plc.PayoffCoilBlock_Expand, (300, text_top + 40), (20, 255, 255))

        # 压辊状态: 1：压下; 0：抬起
        self.__show_text(frame, "jamroll: %d" % self.__plc.JammingRoll_State, (300, text_top + 90), (20, 255, 255))

        # 支撑状态: 1：打开; 0：关闭
        self.__show_text(frame, "support: %d" % self.__plc.support_open, (300, text_top + 140), (20, 255, 255))

        # 导板状态: 1：抬起; 0：放下
        self.__show_text(frame, "board  : %d" % self.__plc.GuideBoard_Up, (300, text_top + 190), (20, 255, 255))

        # 卷筒旋转: 1：正转; 2：反转; 0：不动
        if self.__plc.Payoff_PositiveRun:
            self.__show_text(frame, "block  : 1", (300, text_top + 240), (20, 255, 255))
        elif self.__plc.Payoff_NegativeRun:
            self.__show_text(frame, "block  : 2", (300, text_top + 240), (20, 255, 255))
        else:
            self.__show_text(frame, "block  : 0", (300, text_top + 240), (20, 255, 255))

        # 导板伸缩: 1：伸出; 0：缩回
        self.__show_text(frame, "tongue : %d" % self.__plc.GuideBoard_Extend, (300, text_top + 290), (20, 255, 255))

        # PLC心跳，检测是否与PLC通信正常
        self.__show_text(frame, "PLC heartbeat: %d" % self.__plc.heartbeat, (20, text_top + 340), (20, 255, 255))

        # 内径圆圆心坐标
        ellipse = self.__loader.coil_tracker.coil_ellipse
        if ellipse is not None:
            self.__show_text(frame, "x: %d" % ellipse[0], (20, text_top + 390), (20, 255, 255))
            self.__show_text(frame, "y: %d" % ellipse[1], (300, text_top + 390), (20, 255, 255))
            """
            # 模糊度百分比
            self.__show_text(frame, to_string(D.definition_rate), (2900, 1970), (20, 255, 255))
            self.__show_text(frame, "%", (3000, 1970), (20, 255, 255))
            """

        return text_top + 390 + 50

    def __show_tracking_error(self,frame, text_top):
        status_str = "Tracking error: ellipse error = %5.1f, movement error = %5.1f" % \
                     (self.__loader.coil_tracker.err_ellipse * 100.0, self.__loader.coil_tracker.err_movement * 100.0)
        self.__show_text(frame, status_str, (20, text_top), (20, 255, 255))
        return text_top + 40

    def __show_tracking_info(self, frame, text_top):
        # demo状态下，显示视频总帧数和当前帧
        # if RUN_AS_DEMO:
        #     self.show_text(frame, "Video frame count = %d, current frame = %d" %
        #                    (self.camera.demo_video.frame_cnt, self.camera.demo_video.cur_frame), (20, text_top))
        #     text_top += 40

        # 正在跟踪带卷，显示跟踪状态信息
        status_str = "Tracking status: "
        if self.__loader.coil_tracker.tracking_status == TrackingStatus.TRACKING:
            status_str += "Tracking"
        elif self.__loader.coil_tracker.tracking_status == TrackingStatus.NO_COIL:
            status_str = "No coil"
        elif self.__loader.coil_tracker.tracking_status == TrackingStatus.LOCATING:
            status_str = "Initializing coil center"
            init_center = self.__loader.coil_tracker.coil_locator.center
            if init_center is not None:
                cv2.circle(frame, center=(init_center[0], init_center[1]), radius=5, color=(0, 0, 255), thickness=2)
        self.__show_text(frame, status_str, (20, text_top))
        text_top += 40

        return text_top

    def __show_proc_time(self, frame, text_top):
        loader = self.__loader
        self.__show_text(frame, "Time consumption", (20, text_top))
        self.__show_text(frame, "Total   %4d     Frame:   %4d    Camera:   %4d" %
                         (self.__process_time, self.__frame_time, self.__read_write_frame_time), (20, text_top + 40))
        self.__show_text(frame, "Edge:   %4d     CNN:     %4d" %
                         (self.__edge_time, self.__cnn_time), (20, text_top + 80))
        self.__show_text(frame, "Fit:    %4d     Fit exp: %4d    Offset: %4d" %
                       (self.__fit_time, self.__fit_exp_time, self.__offset_time),
                       (20, text_top + 120))
        self.__show_text(frame, "Track:   %4d    Show:    %4d" %
                         (self.__track_ctrl_time, self.__show_video_time), (20, text_top + 160))
        text_top += 200
        return text_top

    def __show_loading_status(self, frame, text_top):
        status_str = "Loading status: "
        if self.__loader.loading_stage == LoadStage.WAITING_4_LOADING:
            # 在图像上显示带卷中心位置
            status_str += "Waiting to be loaded"
        elif self.__loader.loading_stage == LoadStage.MOVING_TO_UNPACKER:
            status_str += "Moving to the unpacker"
        elif self.__loader.loading_stage == LoadStage.INTO_UNPACKER:
            status_str += "Loading to the unpacker, " + ("loaded" if self.__loader.coil_pos_status.status else "not loaded")
        elif self.__loader.loading_stage == LoadStage.UNPACKING:
            status_str += "Opening the coil, " + "opened" if self.__loader.coil_open_status.status else "not opened"
        self.__show_text(frame, status_str, (20, text_top))
        return text_top + 40

    def run(self):
        self.__event.clear()
        while not self.__terminated:
            if not self.__event.wait(timeout=0.1):
                continue
            self.__event.clear()

            # 将原始灰度图转换为彩色图
            frame_proc = color_image(self.__frame_orig)

            # 显示当前时间
            now = datetime.datetime.now()
            time_str = now.strftime("%Y-%m-%d, %H:%M:%S:") + ("%03d" % int(now.microsecond * 0.001))
            self.__show_text(frame_proc, time_str, (20, 40))

            text_top = 80
            text_top = self.__show_old_loading_info(frame_proc, text_top)
            text_top = self.__show_tracking_error(frame_proc, text_top)
            text_top = self.__show_proc_time(frame_proc, text_top)
            text_top = self.__show_tracking_info(frame_proc, text_top)
            text_top = self.__show_loading_status(frame_proc, text_top)
            text_top = self.__draw_histgram(self.__frame_orig, frame_proc, text_top)
            self.__show_key_operations(frame_proc, text_top)

            self.__draw_lines(frame_proc)
            self.__draw_coil_trace(frame_proc)
            self.__draw_fitting(frame_proc)

            if SCREEN_SHOT_IN_PIT and not RUN_AS_DEMO and \
                    self.__lst_stage == LoadStage.WAITING_4_LOADING and \
                    self.__loader.loading_stage == LoadStage.MOVING_TO_UNPACKER:
                self.__snap_shot = True

            if self.__snap_shot and self.__loader.coil_tracker.coil_ellipse is not None and \
                    self.__loader.coil_tracker.coil_ellipse[1] > 1500 and \
                    self.__loader.coil_tracker.coil_ellipse[0] < 1420:
                cv2.imwrite(self.record_root_path + "\\" + self.__datetime_string() + "-processed.jpg", frame_proc)
                self.__snap_shot = False

            if self.save_record:
                if self.__video_orig is None:
                    self.__init_record(frame_proc)
                self.__video_orig.write(color_image(self.__frame_orig))
                self.__video_proc.write(frame_proc)
                self.__plc.add_csv_record()
            else:
                self.__video_orig = None
                self.__video_proc = None

            # 视频显示
            frame_resized = cv2.resize(frame_proc,
                                       (SCREEN_WIDTH, SCREEN_WIDTH * frame_proc.shape[0] // frame_proc.shape[1]))
            cv2.imshow("control monitor", frame_resized)

            # key operation
            self.key_input = cv2.waitKeyEx(10)

    def show_frame(self, frame_orig):
        self.__frame_orig = frame_orig

        self.__process_time = self.__loader.process_time
        self.__frame_time = self.__loader.frame_time
        self.__read_write_frame_time = self.__loader.read_write_frame_time
        self.__track_ctrl_time = self.__loader.track_ctrl_time
        self.__show_video_time = self.__loader.show_video_time
        self.__edge_time = self.__loader.coil_tracker.edge_time
        self.__cnn_time = self.__loader.coil_tracker.cnn_time
        self.__fit_time = self.__loader.coil_tracker.fit_time
        self.__fit_exp_time = self.__loader.coil_tracker.fit_exp_time
        self.__offset_time = self.__loader.coil_tracker.offset_time

        self.__event.set()

    def close(self):
        self.__terminated = True
        self.join()


class AutoLoader(object):
    def __init__(self, run_as_monitor=False):
        self.run_as_monitor = run_as_monitor

        # 加载各个检测模型
        self.__load_models()
        self.coil_tracker.read_trace_stat("trace_stat.csv")

        # 初始化plc、摄像头硬件，以及显示线程
        self.plc = None
        self.camera = None
        self.show = None
        self.__init_hardwares()

        self.show = ShowThread(loader=self, plc=self.plc)
        self.show.start()

        # 上卷状态机
        self.loading_stage = LoadStage.WAITING_4_LOADING  # 自动上卷的各个阶段
        self.uploading_start_time = time.perf_counter()   # 上卷开始时间，用于在监视模式下，算法出现错误后长时间处于上卷状态
        self.proc_start_time = time.perf_counter()        # 本次控制周期开始时间，用于计算控制周期

        # 各个阶段耗时，统计显示用
        self.frame_time = 0
        self.process_time = 0
        self.read_write_frame_time = 0
        self.track_ctrl_time = 0
        self.show_video_time = 0

        # 试验测试中，如果需要存储每个视频的带卷轨迹数据，则设置save_trace=True
        # 正常现场测试时，默认存储带卷轨迹
        self.csv = None
        self.save_trace = False

        # 保存带卷轨迹跟踪视频
        self.saving_video = False

    def __del__(self):
        if self.plc is not None:
            self.plc.close()
        if self.camera is not None:
            self.camera.close()
        if self.show is not None:
            self.camera.close()

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
    def proc_time(begin_time):
        return int((time.perf_counter() - begin_time) * 1000)

    def control(self):
        self.init_one_cycle()
        self.coil_tracker.restart_tracking()
        self.proc_start_time = time.perf_counter()

        while True:
            self.process_time = self.proc_time(self.proc_start_time)
            self.proc_start_time = time.perf_counter()

            # 控制周期开始时，首先启动读PLC
            self.plc.read()

            # 实际控制时，从摄像头中读取一帧。示例和测试时，从视频文件中读取一帧。
            # 摄像头读取的数据为灰度图，为了显示清晰，将用于显示的图像调整为彩色图
            begin_time = time.perf_counter()
            frame_gray, new_frame, self.frame_time = self.camera.get_one_frame()
            if not new_frame:
                time.sleep(0.01)
                continue
            if frame_gray is None:
                break
            self.read_write_frame_time = self.proc_time(begin_time)

            # 基于当前帧图像和PLC反馈信息，进行带卷跟踪
            begin_time = time.perf_counter()
            self.track_one_frame(frame=frame_gray, plc=self.plc)
            self.car_ctrl()

            # 向PLC写带卷位置
            if self.coil_tracker.coil_ellipse is not None:
                self.plc.copy_coil_ellipse(self.coil_tracker.coil_ellipse)
                # 没有跟踪数据时上传什么??
            # self.plc.write()
            self.track_ctrl_time = self.proc_time(begin_time)

            # 显示控制信息和操作按钮
            begin_time = time.perf_counter()
            self.show.save_record = self.loading_stage != LoadStage.WAITING_4_LOADING
            self.show.show_frame(frame_gray)
            self.show_video_time = self.proc_time(begin_time)

            # 按键操作
            if self.show.key_input in (ord('q'), ord('Q')):
                break

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
        self.csv = None

    def track_one_frame(self, frame, plc):
        # 在监视模式下，上卷终止有可能判断错误，导致始终处于上卷状态
        # 因此，判断上卷开始后，开始计时，如果3分钟内还没有结束，则强制进入等待上卷状态
        if self.loading_stage != LoadStage.WAITING_4_LOADING and self.proc_time(self.uploading_start_time) // 1000 > 600:
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
                    self.uploading_start_time = time.perf_counter()
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

    def test_demo(self, video_file_name):
        # 打开视频文件
        # 如果需要保存轨迹，则建立新的csv文件
        self.camera = CameraThread(video_file_name)
        self.camera.start()
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
