import socket
import time
import datetime
import numpy as np
import cv2
import struct
import csv
import threading
from HKCamera import HKCamera

PLC_IP = "8.8.8.140"
PLC_PORT = 102


class PLCComm(threading.Thread):
    def __init__(self, run_as_demo=False):
        threading.Thread.__init__(self)
        self.run_as_demo = run_as_demo
        self.lock = threading.RLock()
        self.terminated = False
        self.__tcp = None
        self.connected = False
        self.__lst_send_time = datetime.datetime.now()
        self.__csv = None

        # 初始化变量
        self.__init_read_write_buffer()
        self.__init_plc_variables()

    def __init_read_write_buffer(self):
        buf_write = [0x03, 0x00,  # 帧头
                     0x00, 0x5F,  # 数组总长度（95 bytes）
                     0x02, 0xF0, 0x80, 0x32, 0x01, 0x00, 0x00, 0x00, 0x26, 0x00, 0x0E,
                     0x00, 0x40,  # 数据字节的总长度 + 4(即60 + 4 = 64)
                     0x05, 0x01, 0x12, 0x0A, 0x10, 0x02,
                     0x00, 0x3C,  # 数据的总字节数（即60个字节）
                     0x20, 0x12,  # DB号: 2012是DB8210
                     0x84, 0x00,
                     0x00, 0xF0,  # 偏移起始位（F0(H) - 240(D) / 8 = 30.0）
                     0x00, 0x04,
                     0x01, 0xE0,  # 偏移量终止位（即60.0）

                     # 左相机数据
                     0x00, 0x00,  # data[35], data[36]  byte30.0
                     0x00, 0x00,  # data[37], data[38]  byte32.0
                     0x00, 0x00,  # data[39], data[40]  byte34
                     0x00, 0x00,  # data[41], data[42]  byte36
                     0x00, 0x00,  # data[43], data[44]  byte38
                     0x00, 0x00,  # data[45], data[46]  byte40
                     0x00, 0x00,  # data[47], data[48]  byte42
                     0x00, 0x00,  # data[49], data[50]  byte44
                     0x00, 0x00,  # data[51], data[52]  byte46
                     0x00, 0x00,  # data[53], data[54]  byte48
                     0x00, 0x00,  # data[55], data[56]  byte50
                     0x00, 0x00,  # data[55], data[56]  byte52
                     0x00, 0x00,  # data[55], data[56]  byte54
                     0x00, 0x00,  # data[55], data[56]  byte56
                     0x00, 0x00,  # data[55], data[56]  byte58

                     # 右相机数据
                     0xFF, 0xFF,  # data[57], data[58]  byte60.0
                     0x00, 0x00,  # data[59], data[60]  byte62.0
                     0x00, 0x00,  # data[61], data[62]  byte64
                     0x00, 0x00,  # data[63], data[64]  byte66
                     0x00, 0x00,  # data[65], data[66]  byte68
                     0x00, 0x00,  # data[67], data[68]  byte70
                     0x00, 0x00,  # data[69], data[70]  byte72
                     0x00, 0x00,  # data[71], data[72]  byte74
                     0x00, 0x00,  # data[73], data[74]  byte76
                     0x00, 0x00,  # data[75], data[76]  byte78
                     0x00, 0x00,  # data[77], data[78]  byte80
                     0x00, 0x00,  # data[79], data[80]  byte82
                     0x00, 0x00,  # data[81], data[82]  byte84
                     0x00, 0x00,  # data[83], data[84]  byte86
                     0x00, 0x00,  # data[85], data[86]  byte88
                     ]
        self.__buf_write = np.array(buf_write, dtype=np.uint8)

        read_data_key = [0x03, 0x00,
                         0x00, 0x1F,
                         0x02, 0xF0, 0x80, 0x32, 0x01, 0x00, 0x00, 0x03, 0x00, 0x00, 0x0E,
                         0x00, 0x00,
                         0x04, 0x01, 0x12, 0x0A, 0x10, 0x02,
                         0x00, 0x1E,  # 取得数据块的长度, 001E对应30个字节char
                         0x20, 0x12,  # 数据块的地址, 2012对应DB8210
                         0x84, 0x00,
                         0x00, 0x00,  # 起始位
                         ]
        self.__read_data_key = np.array(read_data_key, dtype=np.uint8)
        self.__buf_read = None

    def __init_plc_variables(self):
        # 从PLC中读取到的状态
        self.heartbeat     = 0  # 心跳信号
        self.car_up        = 0  # 开卷机小车上升信号
        self.car_down      = 0  # 开卷机小车下降信号
        self.car_forward   = 0  # 开卷机小车前进信号
        self.car_backward  = 0  # 开卷机小车后退信号
        self.support_open  = 0  # 开卷机小车支撑信号

        # 向PLC写的数据
        self.thread1_heart = 0
        self.thread2_heart = 0

        self.inner_ellipse  = (0, 0, 0, 0, 0)       # 带卷内椭圆
        self.outer_ellipse  = (0, 0, 0, 0, 0)
        self.movement_stage = 0                     # 运动阶段, 0：不运动, 1：上端平移, 2：下降, 3：下端平移
        self.tracking_acc   = 0.0                   # 运动过程中的跟踪精度
        self.surplus_horz   = 0.0                   # 平移阶段的距目的地剩余量
        self.surplus_vert   = 0.0                   # 下降阶段的距目的地剩余量

        self.GuideBoard_Up          = False      # 导板升(1：升, 0：停)
        self.GuideBoard_Down        = False      # 导板降(1：降, 0：停)
        self.GuideBoard_Extend      = False      # 导板伸(1：伸, 0：停)
        self.GuideBoard_Shrink      = False      # 导板缩(1：缩, 0：停)
        self.PayoffCoilBlock_Expand = False      # 卷筒胀径（1：胀径, 0：缩径）
        self.JammingRoll_State      = False      # 压辊压下（1：压下, 0：抬起）
        self.Payoff_PositiveRun     = False      # 卷筒正转（1：正转, 0：停）
        self.Payoff_NegativeRun     = False      # 卷筒反转（1：反转, 0：停）
        self.PayoffSupport_Close    = False      # 程序关闭活动支撑
        self.PayoffSupport_Open     = False      # 程序打开活动支撑

        self.coilload_state         = False      # 上卷状态: 0：未到, 1：到位
        self.coilopen_state         = False      # 开卷状态: 0：未开, 1：开卷
        self.ellipse_track_mode     = False      # 椭圆追踪模式
        self.load_detect_mode       = False      # 上卷检测模式
        self.open_detect_mode       = False      # 开卷检测模式

        self.order_car_up           = False  # 小车手动命令上升
        self.order_car_down         = False  # 小车手动命令下降
        self.order_car_forward      = False  # 小车手动命令前进
        self.order_car_backward     = False  # 小车手动命令后退
        self.order_support_open     = False  # 手动支撑状态

        self.g_get_target_flag      = False    # 是否锁定目标（1：锁定, 0：无目标）
        self.g_AccErr               = False    # 可信度较低报警，当g_acc低于70时，该标志位置1
        self.g_PstErr               = False    # 钢卷位置越界报警，越界时该标志位置1

        self.definition_rate = 0      # 清晰度百分比
        self.definition_flag = True   # 清晰度标志位（1：清晰  0：模糊）

    def __send_tcp(self, order):
        # 频繁向PLC写会造成端口被关闭
        # 检查两次写指令的时间间隔，如果小于10ms，则等待10ms
        # interval = (datetime.datetime.now() - self.__lst_send_time).microseconds
        # self.__lst_send_time = datetime.datetime.now()
        # if interval < 15000:
        #     time.sleep(0.02)
        # time.sleep(0.1)

        try:
            byte_cnt = self.__tcp.send(order)
            if byte_cnt != len(order):
                print(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S:  ") + "向PLC发送数据量不足")
                return False
        except socket.timeout:
            print(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S:  ") + "向PLC发送数据超时")
            return False
        except socket.error as e:
            print(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S:  ") + str(e))
            return False
        return True

    def __read_tcp(self, bytes_cnt):
        try:
            rcved = self.__tcp.recv(bytes_cnt)
            if len(rcved) < bytes_cnt:
                print(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S:  ") + "接收PLC数据量不足，要求%d字节，收到%d字节" % (bytes_cnt, len(rcved)))
                return None
            else:
                return rcved
        except socket.timeout:
            print(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S:  ") + "接收PLC数据超时")
            return None
        except socket.error as e:
            print(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S:  ") + str(e))
            return None

    def connect(self):
        # 连接秘钥1
        datatest1 = [0x03, 0x00, 0x00, 0x16, 0x11, 0xE0, 0x00, 0x00, 0x00, 0x01, 0x00, 0xC1,
                     0x02, 0x01, 0x00, 0xC2, 0x02, 0x01, 0x03, 0xC0, 0x01, 0x0A]
        datatest1 = np.array(datatest1, dtype=np.uint8)

        # 连接秘钥2
        datatest2 = [0x03, 0x00, 0x00, 0x19, 0x02, 0xF0, 0x80, 0x32, 0x01, 0x00, 0x00, 0x01,
                     0x00, 0x00, 0x08, 0x00, 0x00, 0xF0, 0x00, 0x00, 0x01, 0x00, 0x01, 0x03, 0xC0]
        datatest2 = np.array(datatest2, dtype=np.uint8)

        # 创建用于通信的socket
        # 有时候服务器端会中断连接，需要关闭并重新建立连接
        if self.__tcp is not None:
            self.__tcp.shutdown(socket.SHUT_RDWR)
            self.__tcp.close()
            time.sleep(0.5)
        self.__tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

        print(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S:  ") + "正在连接PLC...")
        try:
            self.__tcp.connect((PLC_IP, PLC_PORT))
        except socket.timeout:
            print("连接PLC端口超时")
            return False
        except socket.error as e:
            print(e)
            return False
        self.__tcp.settimeout(0.5)
        print("连接到PLC端口")

        # 向PLC发送密钥1
        time.sleep(0.05)
        if not self.__send_tcp(datatest1):
            print("发送密钥1失败")
            return False

        # 向PLC发送密钥2
        time.sleep(0.05)
        if not self.__send_tcp(datatest2):
            print("发送密钥2失败")
            return False

        print("连接PLC成功")
        self.connected = True
        return True

    def close(self):
        self.terminated = True
        self.join()

    def __send_read_command(self):
        if not self.__send_tcp(self.__read_data_key):
            print("读PLC数据命令发送失败")
            return False
        else:
            return True

    def __receive_from_PLC(self):
        rcved = self.__read_tcp(bytes_cnt=55)
        if rcved is None:
            print("接收PLC数据失败")
            return False
        else:
            self.__buf_read = rcved
            return True

    def __read_uchar(self, id):
        return self.__buf_read[25 + id]

    def __read_short(self, id):
        return (self.__buf_read[25 + id] << 8) | self.__buf_read[26 + id]

    def __read_float(self, id):
        return struct.unpack("!f", (self.__buf_read[25 + id],
                                    self.__buf_read[26 + id],
                                    self.__buf_read[27 + id],
                                    self.__buf_read[28 + id]))[0]

    def __read_bool(self, id, bit):
        return int((self.__buf_read[25 + id] >> bit) & 0x01)

    def __write_short(self, id, value):
        self.__buf_write[35 + id] = (value >> 8) & 0x0FF
        self.__buf_write[36 + id] = value & 0x0FF

    def __write_float(self, id, value):
        # 高位在前，低位在后
        bytes = struct.pack("f", value)
        self.__buf_write[35 + id] = bytes[3]
        self.__buf_write[36 + id] = bytes[2]
        self.__buf_write[37 + id] = bytes[1]
        self.__buf_write[38 + id] = bytes[0]

    def __write_bool(self, id, bit, value):
        mask = 0x01 << bit
        if value:
            self.__buf_write[35 + id] |= mask
        else:
            self.__buf_write[35 + id] &= ~mask

    def new_csv_record(self, file_name):
        csv_file = open(file_name, "w", newline='')
        self.__csv = csv.writer(csv_file)

        row = ["年", "月", "日", "时", "分", "秒", "毫秒",
               "PLC心跳", "点动上", "点动下", "点动进", "点动退",
               "支撑开", "锁定目标", "命令进", "命令退", "命令上", "命令下",
               "椭圆x", "椭圆y", "椭圆h", "椭圆w", "椭圆倾角", "通讯心跳", "图像心跳", "追踪阶段",
               "平移剩余距离", "下降剩余距离", "Rbt报警", "Pst报警", "椭圆追踪模式", "上卷检测模式",
               "开卷检测模式", "上卷状态", "开卷状态", "导板升", "导板降", "导板伸", "导板缩", "卷筒胀径",
               "压辊压下", "卷筒正转", "卷筒反转", "程序关支撑", "程序开支撑"]
        self.__csv.writerow(row)

    def add_csv_record(self):
        if self.__csv is None:
            return
        now = datetime.datetime.now()
        row = [str(now.year), str(now.month), str(now.day),
               str(now.hour), str(now.minute), str(now.second), str(int(now.microsecond * 0.001)),
               str(self.heartbeat),
               str(self.car_up), str(self.car_down), str(self.car_forward), str(self.car_backward),
               str(self.support_open),
               str(self.g_get_target_flag),
               str(self.order_car_forward),
               str(self.order_car_backward),
               str(self.order_car_up),
               str(self.order_car_down),
               str(self.inner_ellipse[0]),
               str(self.inner_ellipse[1]),
               str(self.inner_ellipse[2]),
               str(self.inner_ellipse[3]),
               str(self.inner_ellipse[4]),
               str(self.thread1_heart),
               str(self.thread2_heart),
               str(self.movement_stage),
               str(self.surplus_horz),
               str(self.surplus_vert),
               str(self.g_AccErr),
               str(self.g_PstErr),
               str(self.ellipse_track_mode),
               str(self.load_detect_mode),
               str(self.open_detect_mode),
               str(self.coilload_state),
               str(self.coilopen_state),
               str(self.GuideBoard_Up),
               str(self.GuideBoard_Down),
               str(self.GuideBoard_Extend),
               str(self.GuideBoard_Shrink),
               str(self.PayoffCoilBlock_Expand),
               str(self.JammingRoll_State),
               str(self.Payoff_PositiveRun),
               str(self.Payoff_NegativeRun),
               str(self.PayoffSupport_Close),
               str(self.PayoffSupport_Open)]
        self.__csv.writerow(row)

    def copy_coil_ellipse(self, ellipse):
        self.lock.acquire()
        self.inner_ellipse = ellipse
        self.lock.release()

    def refresh_status(self):
        """
        从PLC中读取上卷机和开卷机等的当前状态
        :return: 数据更新成功返回True，否则返回False
        """
        if self.__send_read_command() and self.__receive_from_PLC():
            # 根据张方远的程序，从PLC中读取的数据仅包含小车运动信号
            self.heartbeat    = self.__read_short(0)
            self.car_up       = self.__read_bool(29, 0)       # 开卷机小车上升信号
            self.car_down     = self.__read_bool(29, 1)       # 开卷机小车下降信号
            self.car_forward  = self.__read_bool(29, 2)       # 开卷机小车前进信号
            self.car_backward = self.__read_bool(29, 3)       # 开卷机小车后退信号
            self.support_open = self.__read_bool(29, 4)       # 开卷机支撑信号
            return True
        else:
            return False

    def send_order(self):
        self.lock.acquire()

        self.thread1_heart += 1
        self.thread2_heart += 1
        if self.thread1_heart > 5000:
            self.thread1_heart = 0
            self.thread2_heart = 0

        self.__write_short(0, self.thread1_heart)                     # 数据地址30.0   线程1心跳
        self.__write_float(2, float(self.thread2_heart))              # 数据地址32.0，图像处理周期 / 图像处理线程心跳
        self.__write_float(6, float(self.inner_ellipse[0]))           # 数据地址36.0，椭圆中心x
        self.__write_float(10, float(self.inner_ellipse[1]))          # 数据地址40.0，椭圆中心y
        self.__write_float(14, float(self.inner_ellipse[2]))          # 数据地址44.0，椭圆长轴
        self.__write_float(18, float(self.inner_ellipse[3]))          # 数据地址48.0，椭圆短轴
        self.__write_float(22, float(self.movement_stage))            # 数据地址52.0，上卷阶段（0：无目标 1：平移  2：下降  3：对准）
        self.__write_float(26, float(self.tracking_acc))              # 数据地址56.0，可信度 %
        self.__write_float(30, float(self.inner_ellipse[4]))          # 数据地址60.0，椭圆倾斜角
        self.__write_float(34, float(self.surplus_horz))              # 数据地址64.0，平移阶段的剩余距离（mm）
        self.__write_float(38, float(self.surplus_vert))              # 数据地址68.0，下降阶段的剩余距离（mm）
        self.__write_float(42, float(self.definition_rate))           # 数据地址72.0，模糊程度百分比

        self.__write_bool(56, 0, self.GuideBoard_Up)                  # 数据地址86.0，导板升(1：升  0：停)
        self.__write_bool(56, 1, self.GuideBoard_Down)                # 数据地址86.1，导板降(1：降  0：停)
        self.__write_bool(56, 2, self.GuideBoard_Extend)              # 数据地址86.2，导板伸(1：伸  0：停)
        self.__write_bool(56, 3, self.GuideBoard_Shrink)              # 数据地址86.3，导板缩(1：缩  0：停)
        self.__write_bool(56, 4, self.PayoffCoilBlock_Expand)         # 数据地址86.4，卷筒胀径（1：胀径  0：缩径）
        self.__write_bool(56, 5, self.JammingRoll_State)              # 数据地址86.5，压辊压下（1：压下  0：抬起）
        self.__write_bool(56, 6, self.Payoff_PositiveRun)             # 数据地址86.6，卷筒正转（1：正转 0：停）
        self.__write_bool(56, 7, self.Payoff_NegativeRun)             # 数据地址86.7，卷筒反转（1：反转 0：停）

        self.__write_bool(57, 0, self.PayoffSupport_Close)            # 数据地址87.0，程序关闭活动支撑
        self.__write_bool(57, 1, self.PayoffSupport_Open)             # 数据地址87.1，程序打开活动支撑
        self.__write_bool(57, 2, self.coilload_state)                 # 数据地址87.2，上卷状态  0：未上卷  1：已上卷
        self.__write_bool(57, 3, self.coilopen_state)                 # 数据地址87.3，开卷状态  0：未开卷  1：已开卷
        self.__write_bool(57, 4, self.ellipse_track_mode)             # 数据地址87.4，椭圆追踪
        self.__write_bool(57, 5, self.load_detect_mode)               # 数据地址87.5，上卷检测
        self.__write_bool(57, 6, self.open_detect_mode)               # 数据地址87.6，开卷检测
        self.__write_bool(57, 7, self.definition_flag)                # 数据地址87.7，模糊度标志

        self.__write_bool(58, 1, self.order_car_up)                   # 数据地址88.1，写本地命令，上卷小车上升
        self.__write_bool(58, 2, self.order_car_down)                 # 数据地址88.2，写本地命令，上卷小车下降
        self.__write_bool(58, 3, self.order_car_forward)              # 数据地址88.3，写本地命令，上卷小车前进
        self.__write_bool(58, 4, self.order_car_backward)             # 数据地址88.4，写本地命令，上卷小车后退

        self.__write_bool(58, 5, self.g_get_target_flag)              # 数据地址88.5，是否锁定目标（1：锁定  0：无目标）
        self.__write_bool(58, 6, self.g_AccErr)                       # 数据地址88.6，可信度较低报警，当可信度低于70时，该标志位置1
        self.__write_bool(58, 7, self.g_PstErr)                       # 数据地址88.7，钢卷位置越界报警，越界时该标志位置1

        self.lock.release()

        return self.__send_tcp(self.__buf_write)

    def run(self):
        if self.run_as_demo:
            while not self.terminated:
                time.sleep(0.02)
        else:
            self.connect()
            while not self.terminated:
                time.sleep(0.1)
                self.refresh_status()
                time.sleep(0.1)
                if not self.send_order():
                    self.connect()


if __name__ == '__main__':
    plc = PLCComm()
    plc.start()


    """
    camera = HKCamera()
    SCREEN_WIDTH = 1400

    if plc.connect():
        time.sleep(3)
        while True:
            plc.refresh_status()
            print((plc.heartbeat, plc.car_up, plc.car_down, plc.car_forward, plc.car_backward))

            img = camera.get_image()
            img = cv2.resize(img, (SCREEN_WIDTH, int(SCREEN_WIDTH * img.shape[0] / img.shape[1])))
            cv2.imshow("video", img)
            key = cv2.waitKeyEx(300)

            plc.order_car_up = False
            plc.order_car_down = False
            plc.order_car_forward = False
            plc.order_car_backward = False
            if key == 2490368:    # up
                plc.order_car_up = True
            elif key == 2621440:
                plc.order_car_down = True
            elif key == 2424832:
                plc.order_car_forward = True
            elif key == 2555904:
                plc.order_car_backward = True
            elif key == ord('q'):
                break
    """
