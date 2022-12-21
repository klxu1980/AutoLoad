import cv2
import math
from CoilNet import CoilNet
from CoilTracer import CoilTracer
from ImageSetEx import ImageSet


class TopTracer(CoilTracer):
    def __init__(self):
        image_size = 256
        self.WINDOW_SIZE = image_size

        network = CoilNet(img_size=image_size, output_size=2)
        network.add_conv_layer(channels=8, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=16, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=32, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=64, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=128, kernel_size=3, padding=1, pool_size=2)
        network.add_conv_layer(channels=256, kernel_size=3, padding=1, pool_size=2)
        network.add_linear_layer_by_divider(divider=32)
        network.add_linear_layer(output_size=2)
        CoilTracer.__init__(self, network=network, output_size=2)

        # 使用CNN确定中心位置时，在初始位置附近，随机产生一系列窗口，对窗口中的图像检测中心，然后求平均
        self.net_x_range = (-20, 20)          # 随机窗口x方向范围
        self.net_y_range = (-20, 20)          # 随机窗口y方向范围
        self.net_count = 8                   # 随机窗口数量

        self.expected_trace = None
        self.error_max = 5

        self.pre_proc = True

    def analyze_one_frame(self, frame):
        # 计算轨迹
        center = self.center_by_net(frame, img_size=self.WINDOW_SIZE, init_center=self.init_center)
        center = (int(center[0]), int(center[1]))
        self.init_center = center

        # 判断位置高低
        status = 0
        if self.expected_trace is not None:
            line = self.expected_trace
            exp_y = (line[1] - line[3]) * (center[0] - line[2]) / (line[0] - line[2]) + line[3]
            if math.fabs(center[1] - exp_y) > self.error_max:
                status = 1 if center[1] < exp_y else -1

        return center, status

    def show_trace(self, frame, center, color):
        if self.expected_trace is not None:
            cv2.line(frame, (self.expected_trace[0], self.expected_trace[1]),
                     (self.expected_trace[2], self.expected_trace[3]), (20, 255, 255), 1)
            cv2.line(frame, (self.expected_trace[0], self.expected_trace[1] + self.error_max),
                     (self.expected_trace[2], self.expected_trace[3] + self.error_max), (20, 20, 255), 1)
            cv2.line(frame, (self.expected_trace[0], self.expected_trace[1] - self.error_max),
                     (self.expected_trace[2], self.expected_trace[3] - self.error_max), (20, 20, 255), 1)

        super().show_trace(frame, center, color)


if __name__ == '__main__':
    Net = TopTracer().coil_net
    Net.learn_rate = 0.00001
    Net.l2_lambda = 0.0002
    Net.dropout_rate = 0.4
    Net.save_dir = ""
    Net.save_prefix = "带卷左上角"
    Net.save_interval = 10000

    mode = input("Train(T) the model, or run(R) the model?")
    sample_dir = "E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷对准\\带卷对准训练"
    if mode == 't' or mode == 'T':
        train_set = ImageSet(sample_dir, img_size=Net.img_size, grey_scaled=False, pre_proc=True)
        eval_set = ImageSet(sample_dir, img_size=Net.img_size, grey_scaled=False, pre_proc=True)

        train_set.regen_count = 20
        train_set.regen_interval = 500
        eval_set.regen_count = 20
        eval_set.regen_interval = 500

        Net.train(train_set, eval_set, epoch_cnt=10000, mini_batch=64)
    else:
        Net.load_model("带卷左上角2022-07-25-22-11.pt")
        eval_set = ImageSet(sample_dir, img_size=Net.img_size)
        Net.test_net(eval_set)