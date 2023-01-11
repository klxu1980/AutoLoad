import socket
import time
import numpy as np

PLC_IP = "8.8.8.140"
PLC_PORT = 102


class PLCComm(object):
    def __init__(self):
        buf_write = [0x03, 0x00,    # 帧头
                          0x00, 0x5F,    # 数组总长度（95 bytes）
                          0x02, 0xF0, 0x80, 0x32, 0x01, 0x00, 0x00, 0x00, 0x26, 0x00, 0x0E,
                          0x00, 0x40,    # 数据字节的总长度 + 4(即60 + 4 = 64)
                          0x05, 0x01, 0x12, 0x0A, 0x10, 0x02,
                          0x00, 0x3C,    # 数据的总字节数（即60个字节）
                          0x20, 0x12,    # DB号: 2012是DB8210
                          0x84, 0x00,
                          0x00, 0xF0,    # 偏移起始位（F0(H) - 240(D) / 8 = 30.0）
                          0x00, 0x04,
                          0x01, 0xE0,    # 偏移量终止位（即60.0）

                          # 左相机数据
                          0x00, 0x00,    # data[35], data[36]  byte30.0
                          0x00, 0x00,    # data[37], data[38]  byte32.0
                          0x00, 0x00,    # data[39], data[40]  byte34
                          0x00, 0x00,    # data[41], data[42]  byte36
                          0x00, 0x00,    # data[43], data[44]  byte38
                          0x00, 0x00,    # data[45], data[46]  byte40
                          0x00, 0x00,    # data[47], data[48]  byte42
                          0x00, 0x00,    # data[49], data[50]  byte44
                          0x00, 0x00,    # data[51], data[52]  byte46
                          0x00, 0x00,    # data[53], data[54]  byte48
                          0x00, 0x00,    # data[55], data[56]  byte50
                          0x00, 0x00,    # data[55], data[56]  byte52
                          0x00, 0x00,    # data[55], data[56]  byte54
                          0x00, 0x00,    # data[55], data[56]  byte56
                          0x00, 0x00,    # data[55], data[56]  byte58

                          # 右相机数据
                          0xFF, 0xFF,   # data[57], data[58]  byte60.0
                          0x00, 0x00,   # data[59], data[60]  byte62.0
                          0x00, 0x00,   # data[61], data[62]  byte64
                          0x00, 0x00,   # data[63], data[64]  byte66
                          0x00, 0x00,   # data[65], data[66]  byte68
                          0x00, 0x00, # data[67], data[68]  byte70
                          0x00, 0x00, # data[69], data[70]  byte72
                          0x00, 0x00, # data[71], data[72]  byte74
                          0x00, 0x00, # data[73], data[74]  byte76
                          0x00, 0x00, # data[75], data[76]  byte78
                          0x00, 0x00, # data[77], data[78]  byte80
                          0x00, 0x00, # data[79], data[80]  byte82
                          0x00, 0x00, # data[81], data[82]  byte84
                          0x00, 0x00, # data[83], data[84]  byte86
                          0x00, 0x00, # data[85], data[86]  byte88
                          ]
        self.buf_write = np.array(buf_write, dtype=np.uint8)

        read_data_key = [0x03, 0x00,
                            0x00, 0x1F,
                            0x02, 0xF0, 0x80, 0x32, 0x01, 0x00, 0x00, 0x03, 0x00, 0x00, 0x0E,
                            0x00, 0x00,
                            0x04, 0x01, 0x12, 0x0A, 0x10, 0x02,
                            0x00, 0x1E, # 取得数据块的长度, 001E对应30个字节char
                            0x20, 0x12, # 数据块的地址, 2012对应DB8210
                            0x84, 0x00,
                            0x00, 0x00, # 起始位
                            ]
        self.read_data_key = np.array(read_data_key, dtype=np.uint8)

        self.buf_read = np.zeros(100, dtype=np.uint8)   # 用于接收数据缓冲区

        # 从PLC中读取到的状态
        self.heartbeat    = 0  # 心跳信号
        self.car_up       = 0  # 开卷机小车上升信号
        self.car_down     = 0  # 开卷机小车下降信号
        self.car_forward  = 0  # 开卷机小车前进信号
        self.car_backward = 0  # 开卷机小车后退信号
        self.support_open = 0  # 开卷机小车支撑信号

        # TCP
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def read_tcp(self):
        pass

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
        print("正在连接PLC...")
        self.tcp.connect((PLC_IP, PLC_PORT))
        self.tcp.settimeout(0.5)
        print("连接到PLC端口")

        time.sleep(secs=0.05)
        ret = self.tcp.send(datatest1)
        if not ret:
            print("秘钥1发送失败")
            return False

        time.sleep(secs=0.05)
        ret = self.tcp.send(datatest2)
        if not ret:
            print("秘钥2发送失败")
            return False

        print("连接PLC成功")
        return True

    def send_read_command(self):
        if not self.tcp.send(self.read_data_key):
            print("读数据命令发送失败")
            return False
        else:
            return True

    def receive_from_PLC(self):
        self.receive_from_PLC()
        return True

    def read_uchar(self, id):
        return self.buf_read[25 + id]

    def read_short(self, id):
        return (self.buf_read[25 + id] << 8) | self.buf_read[26 + id]

    def read_float(self, id):
        pass

    def read_bool(self, id, bit):
        return (self.buf_read[25 + id] >> bit) & 0x01

    def write_short(self, id, value):
        self.buf_write[35 + id] = value >> 8
        self.buf_write[36 + id] = value

    def write_float(self, id, value):
        pass

    def write_bool(self, id, bit, value):
        mask = 0x01 << bit
        if value:
            self.buf_write[35 + id] |= mask
        else:
            self.buf_write[35 + id] &= ~mask

    def read_status(self):
        if self.send_read_command() and self.receive_from_PLC():
            self.heartbeat    = self.read_short(0)
            self.car_up       = self.read_bool(29, 0)       # 开卷机小车上升信号
            self.car_down     = self.read_bool(29, 1)       # 开卷机小车下降信号
            self.car_forward  = self.read_bool(29, 2)       # 开卷机小车前进信号
            self.car_backward = self.read_bool(29, 3)       # 开卷机小车后退信号
            self.support_open = self.read_bool(29, 4)       # 开卷机支撑信号


if __name__ == '__main__':
    plc = PLCComm()
    if plc.connect():
       while True:
           time.sleep(0.5)
           plc.read_status()
           print("小车上升： %d, 小车下降： %d, 小车前进： %d, 小车后退： %d, 开卷机支撑： %d" %
                 (plc.car_up, plc.car_down, plc.car_forward, plc.car_backward))