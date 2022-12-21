import cv2
import time
import os
import glob
import datetime
import csv
import numpy as np
from enum import Enum
from CenterTracer import CenterTracer
from TopTracer import TopTracer
from CoilLocate import CoilLocator
from ImageSet import LabelType
from CoilStatus import CoilOpenStatus
from CoilStatus import CoilPosStatus


AIMING_BEGIN = (1450, 1450)                      # 带卷对准开卷机的检测开始和停止位置
AIMING_END = 1250

SCREEN_WIDTH = 1400
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
    WAITING_4_LOADING = 0,   # 在堆料区等待上料
    DOWNWARD_MOVING   = 1,   # 向基坑中移动
    AIMING_UNPACKER   = 2,   # 对准开卷机运动
    INTO_UNPACKER     = 3,   # 进入开卷机(判断是否到达开卷位置)
    UNPACKING         = 4,   # 开卷(判断是否开卷)
    UPLOAD_COMPLETED  = 5    # 开卷完成


class AimingStatus(Enum):
    OK       = 0,
    TOO_LOW  = -1,
    TOO_HIGH = 1


class AutoLoader(object):
    def __init__(self):
        # self.coil_tracer是核心跟踪器
        self.coil_tracer = CenterTracer()
        self.coil_tracer.load_model("带卷定位_有干扰_2022-08-05-16-05.pt")

        # 带卷初始位置检测器
        self.coil_locator = CoilLocator()
        self.coil_locator.load_model("带卷初始位置2022-08-18-22-55.pt")

        # 上卷检测器
        self.coil_pos_status = CoilPosStatus(src_size=1200, src_position=(0, 800))
        self.coil_pos_status.init_model()
        self.coil_pos_status.load_model("上卷检测2020-10-24-16-30.pt")

        # 开卷检测器
        self.coil_open_status = CoilOpenStatus(src_size=1200, src_position=(0, 800))
        self.coil_open_status.init_model()
        self.coil_open_status.load_model("开卷检测2020-10-24-12-12.pt")

        # 上卷各个阶段
        self.loading_stage = LoadStage.WAITING_4_LOADING
        self.unpacker_aiming_begin = (1450, 1450)  # 带卷对准开卷机的检测开始和停止位置
        self.unpacker_aiming_end = 1250

        self.aiming_ok_line = None
        self.aiming_low_line = None
        self.aiming_high_line = None

        self.proc_time = 0.0

        self.video = None
        self.csv = None
        self.save_trace = False

        self.saving_video = False
        self.video_save = None
        self.paused = False
        self.coil_top = None

        self.load_aiming_lines()

    @staticmethod
    def datetime_string():
        now = datetime.datetime.now()
        datetime_str = "%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        return datetime_str

    @staticmethod
    def show_text(image, text, pos, color=(0, 255, 0)):
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    def load_aiming_lines(self):
        csv_reader = csv.reader(open("trace_stat_正常.csv", 'r'))
        for i, line in enumerate(csv_reader):
            if i == 0:
                self.aiming_ok_line = (float(line[0]), float(line[1]))
                break

        csv_reader = csv.reader(open("trace_stat_偏高.csv", 'r'))
        for i, line in enumerate(csv_reader):
            if i == 1:
                self.aiming_high_line = (float(line[0]), float(line[1]))
                break

        csv_reader = csv.reader(open("trace_stat_偏低.csv", 'r'))
        for i, line in enumerate(csv_reader):
            if i == 2:
                self.aiming_low_line = (float(line[0]), float(line[1]))
                break

    def init_one_test_cycle(self, file_name):
        self.video = VideoPlayer(file_name)
        self.coil_locator.init_one_cycle()

        self.loading_stage = 0
        self.saving_video = False
        self.paused = False

        if self.save_trace:
            csv_file = open(self.video.file_name + "_new.csv", "w", newline='')
            self.csv = csv.writer(csv_file)

    def save_video(self, frame):
        if self.saving_video:
            if self.video_save is None:
                datetime_str = self.datetime_string()
                size = (frame.shape[1], frame.shape[0])
                #self.video_save = cv2.VideoWriter("d:\\" + datetime_str + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 10, size)
                self.video_save = cv2.VideoWriter("d:\\" + datetime_str + ".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 10,
                                                  size)
            self.video_save.write(frame)
            self.show_text(frame, "recording video", pos=(600, 40), color=(0, 0, 255))
        else:
            self.video_save = None

    def save_ellipse(self, ellipse):
        row = ["%d" % value for value in ellipse]
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
        """
        cv2.line(image,
                 (x_begin, int(x_begin * self.aiming_ok[0] + self.aiming_ok[1])),
                 (x_end, int(x_end * self.aiming_ok[0] + self.aiming_ok[1])),
                 color=(0, 255, 255), thickness=2)
        """
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
        self.coil_locator.init_one_cycle()
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

    def is_coil_into_unpacker(self, frame):
        """
        检查带卷是否已经插入开卷机就位
        :param frame: 由摄像头获得的带卷RGB图像
        :return: True: 已插入开卷机；False: 未插入开卷机
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        status = self.coil_pos_status.analyze_one_frame(frame)
        return status

    def is_coil_unpacked(self, frame):
        """
        检查带卷是否已经开卷
        :param frame: 由摄像头获得的带卷RGB图像
        :return: True: 已开卷；False: 未开卷
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        status = self.coil_open_status.analyze_one_frame(frame)
        return status

    def control_demo(self, frame):
        start_time = time.perf_counter()
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            # 当带卷位于堆料区等待上卷时，持续定位其初始坐标(并滤波)，将初始位置写入跟踪器初始位置中
            coil_center = self.coil_locator.center_by_net(frame)
            init_ellipse = (coil_center[0], coil_center[1], 100, 100, 0)  # 椭圆初始信息仅包含中心坐标，长短轴和角度忽略(随便填数)
            self.coil_tracer.init_one_cycle(init_ellipse=init_ellipse)
        elif self.loading_stage == LoadStage.DOWNWARD_MOVING:
            # 带卷向基坑移动
            # 当带卷到达基坑底部后，转为对准开卷机
            ellipse = self.coil_tracer.analyze_one_frame(frame)  # obtain the center
            if ellipse[0] < self.unpacker_aiming_begin[0] and ellipse[1] > self.unpacker_aiming_begin[1]:
                self.loading_stage = LoadStage.AIMING_UNPACKER
        elif self.loading_stage == LoadStage.AIMING_UNPACKER:
            # 带卷瞄准开卷机，此时需要评估带卷中心点是否在允许范围内
            # 当带卷套入开卷机后，状态转为插入开卷机
            ellipse = self.coil_tracer.analyze_one_frame(frame)  # obtain the center
            aiming_status = self.check_aiming_status(ellipse)
            if ellipse[0] < self.unpacker_aiming_end:
                self.loading_stage = LoadStage.INTO_UNPACKER
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            status = self.coil_pos_status.analyze_one_frame(frame)
            if status:
                self.loading_stage = LoadStage.UNPACKING
        elif self.loading_stage == LoadStage.UNPACKING:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            status = self.coil_open_status.analyze_one_frame(frame)
            if status:
                self.loading_stage = LoadStage.UPLOAD_COMPLETED

        self.proc_time = (time.perf_counter() - start_time) * 1000


    def control(self, frame):
        """
        :param frame: 输入图像为RGB彩色图
        :return:
        """
        start_time = time.perf_counter()
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            # 当带卷位于堆料区等待上卷时，持续定位其初始坐标(并滤波)，将初始位置写入跟踪器初始位置中
            coil_center = self.coil_locator.center_by_net(frame)
            init_ellipse = (coil_center[0], coil_center[1], 100, 100, 0)  # 椭圆初始信息仅包含中心坐标，长短轴和角度忽略(随便填数)
            self.coil_tracer.init_one_cycle(init_ellipse=init_ellipse)
        elif self.loading_stage == LoadStage.DOWNWARD_MOVING:
            # 带卷向基坑移动
            # 当带卷到达基坑底部后，转为对准开卷机
            ellipse = self.coil_tracer.analyze_one_frame(frame)  # obtain the center
            if ellipse[0] < self.unpacker_aiming_begin[0] and ellipse[1] > self.unpacker_aiming_begin[1]:
                self.loading_stage = LoadStage.AIMING_UNPACKER
        elif self.loading_stage == LoadStage.AIMING_UNPACKER:
            # 带卷瞄准开卷机，此时需要评估带卷中心点是否在允许范围内
            # 当带卷套入开卷机后，状态转为插入开卷机
            ellipse = self.coil_tracer.analyze_one_frame(frame)  # obtain the center
            aiming_status = self.check_aiming_status(ellipse)
            if ellipse[0] < self.unpacker_aiming_end:
                self.loading_stage = LoadStage.INTO_UNPACKER
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            status = self.coil_pos_status.analyze_one_frame(frame)
            if status:
                self.loading_stage = LoadStage.UNPACKING
        elif self.loading_stage == LoadStage.UNPACKING:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            status = self.coil_open_status.analyze_one_frame(frame)
            if status:
                self.loading_stage = LoadStage.UPLOAD_COMPLETED

        self.proc_time = (time.perf_counter() - start_time) * 1000

    def show_loading_info(self, frame):
        if self.loading_stage == LoadStage.WAITING_4_LOADING:
            # 在图像上显示带卷中心位置
            init_center = self.coil_locator.center.astype(np.int32)
            cv2.circle(frame, center=init_center, radius=5, color=(0, 0, 255), thickness=2)
        elif self.loading_stage == LoadStage.DOWNWARD_MOVING or self.loading_stage == LoadStage.AIMING_UNPACKER:
            self.draw_aiming_lines(frame)
            self.coil_tracer.show_trace(frame)

            ellipse = self.coil_tracer.coil_ellipse
            status_str = "Coil center = (%d, %d), err(fit/exp) = %.2f, err(cnn/exp) = %.2f, err(cnn/fit) = %.2f" % \
                         (ellipse[0], ellipse[1],
                          self.coil_tracer.err_fit_vs_exp * 100.0,
                          self.coil_tracer.err_cnn_vs_exp * 100.0,
                          self.coil_tracer.err_cnn_vs_fit * 100.0)

            if self.loading_stage == LoadStage.AIMING_UNPACKER:
                aiming_status = ("position ok", "position low", "position high")
                status_str += ", " + aiming_status[self.check_aiming_status(ellipse)]

            self.show_text(frame, status_str, (20, 40))
        elif self.loading_stage == LoadStage.INTO_UNPACKER:
            status = self.coil_pos_status.status
            self.show_text(frame, "coil loaded" if status else "coil not loaded", (20, 40))
        elif self.loading_stage == LoadStage.UNPACKING:
            status = self.coil_open_status.status
            self.show_text(frame, "coil opened" if status else "coil not opened", (20, 40))

        # 显示计算耗时和当前帧
        self.show_text(frame, "computing time = %6.2f ms" % self.proc_time, (20, 80))

        return frame

    def key_operate(self, frame):
        # 显示操作按键
        self.show_key_operations(frame, row_begin=200)

        # 视频显示
        frame_resized = cv2.resize(frame, (SCREEN_WIDTH, int(SCREEN_WIDTH * frame.shape[0] / frame.shape[1])))
        cv2.imshow("video", frame_resized)

        key = cv2.waitKeyEx(0 if self.paused else 10)
        if key == ord('p') or key == ord('P'):
            self.paused = not self.paused
        elif key == 2555904:
            if self.paused:
                self.video.forward()
        elif key == 2424832:
            if self.paused:
                self.video.backward()
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
            self.video.forward()

    def test2(self):
        self.init_one_cycle()
        while self.video is not None and not self.video.is_end():
            # 从摄像头或文件中读取帧，调整尺寸并灰度化
            frame = self.video.get_frame()
            frame_4_proc = frame.copy()

            self.control(frame=frame_4_proc)
            frame = self.show_loading_info(frame)

            self.key_operate(frame)

            if not self.paused:
                self.video.forward()

    def test(self, init_pos=None):
        while self.video is not None and not self.video.is_end():
            start_time = time.perf_counter()

            # 从摄像头或文件中读取帧，调整尺寸并灰度化
            frame = self.video.get_frame()
            frame_4_proc = frame.copy()

            self.draw_aiming_lines(frame)

            if self.loading_stage == 0:
                if init_pos is None:
                    coil_center = self.coil_locator.center_by_net(frame_4_proc)
                    if self.coil_locator.proc_times > 10:
                        init_ellipse = (coil_center[0], coil_center[1], 100, 100, 0)
                        self.coil_tracer.init_one_cycle(init_ellipse=init_ellipse)
                        self.loading_stage = 1
                else:
                    init_ellipse = (init_pos[0], init_pos[1], 100, 100, 0)
                    self.coil_tracer.init_one_cycle(init_ellipse=init_ellipse)
                    self.loading_stage = 1
            elif self.loading_stage == 1:
                ellipse = self.coil_tracer.analyze_one_frame(frame_4_proc)   # obtain the center
                self.coil_tracer.show_trace(frame)
                status_str = "Coil center = (%d, %d), err(fit/exp) = %.2f, err(cnn/exp) = %.2f, err(cnn/fit) = %.2f" % \
                             (ellipse[0], ellipse[1],
                                self.coil_tracer.err_fit_vs_exp * 100.0,
                                self.coil_tracer.err_cnn_vs_exp * 100.0,
                                self.coil_tracer.err_cnn_vs_fit * 100.0)

                if ellipse[0] < AIMING_BEGIN[0] and ellipse[1] > AIMING_BEGIN[1]:
                    aiming_status = self.check_aiming_status(ellipse)
                    if aiming_status == -1:
                        status_str += ", position low"
                    elif aiming_status == 1:
                        status_str += ", position high"
                    else:
                        status_str += ", position ok"

                self.show_text(frame, status_str, (20, 40))

                if self.save_trace and ellipse[0] < 1800:
                    self.save_ellipse(ellipse)
                if ellipse[0] < AIMING_END:
                    self.loading_stage = 2
            elif self.loading_stage == 2:
                frame_4_proc = cv2.cvtColor(frame_4_proc, cv2.COLOR_BGR2GRAY)
                status = self.coil_pos_status.analyze_one_frame(frame_4_proc)
                self.show_text(frame, "coil loaded" if status else "coil not loaded", (20, 40))

                if status:
                    self.loading_stage = 3
            elif self.loading_stage == 3:
                frame_4_proc = cv2.cvtColor(frame_4_proc, cv2.COLOR_BGR2GRAY)
                status = self.coil_open_status.analyze_one_frame(frame_4_proc)
                self.show_text(frame, "coil opened" if status else "coil not opened", (20, 40))

            # 显示计算耗时和当前帧
            elapsed = (time.perf_counter() - start_time) * 1000
            self.show_text(frame, "computing time = %6.2f ms" % elapsed, (20, 80))
            self.show_text(frame, "frame cnout = %d, current frame = %d" % (self.video.frame_cnt, self.video.cur_frame),
                           (20, 140))

            # 显示操作按键
            self.show_key_operations(frame, row_begin=200)

            # 视频显示
            frame_resized = cv2.resize(frame, (SCREEN_WIDTH, int(SCREEN_WIDTH * frame.shape[0] / frame.shape[1])))
            cv2.imshow("video", frame_resized)

            # 保存视频
            self.save_video(frame_resized)

            # key operation
            key = cv2.waitKeyEx(0 if self.paused else 10)
            if key == ord('p') or key == ord('P'):
                self.paused = not self.paused
            elif key == 2555904:
                if self.paused:
                    self.video.forward()
            elif key == 2424832:
                if self.paused:
                    self.video.backward()
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
                self.video.forward()


def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files)


if __name__ == '__main__':
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
        os.chdir("G:\\01 自动上卷视频\\内椭圆参数统计视频")
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
        for file_name in glob.glob("*.avi"):
            auto_loader.init_one_test_cycle(file_name)
            auto_loader.test(init_pos)
    else:
        auto_loader.save_trace = True
        for root, dirs, files in os.walk("G:\\01 自动上卷视频"):
            for _, file in enumerate(files):
                if os.path.splitext(file)[-1] == ".avi":
                    file_path = root + "\\" + file
                    auto_loader.init_one_test_cycle(file_path)
                    auto_loader.test()
