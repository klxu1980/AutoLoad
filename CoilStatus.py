import torch
import cv2
import numpy as np
from BaseImageNet import BaseImageNet
from CoilImgs import CoilImageSet


class CoilStatusNet(BaseImageNet):
    def __init__(self, status_cnt, src_size, src_position, dst_size):
        super(CoilStatusNet, self).__init__(regress=False, img_size=dst_size)
        self.status_cnt = status_cnt
        self.src_size = src_size
        self.src_position = src_position
        self.dst_size = dst_size
        self.status = 0

    def predict(self, batch_img):
        outputs = super().predict(batch_img)
        return np.argmax(outputs[0])

    def train(self, train_set, eval_set, epoch_cnt, mini_batch):
        super().train(train_set, eval_set, epoch_cnt, mini_batch)

    def test_net(self, eval_set):
        batch_img, batch_label, raw_img = eval_set.random_sample(1)

        output = self.predict(batch_img)
        img = raw_img[0]
        print("label = %d, predict = %d" % (batch_label[0], output[0]))
        cv2.imshow("", img)

    def analyze_one_frame(self, frame):
        img = frame[self.src_position[1]: self.src_position[1] + self.src_size,
                      self.src_position[0]: self.src_position[0] + self.src_size].copy()
        img = cv2.resize(img, (self.img_size, self.img_size))
        norm_img = self.normalize_image(img)

        batch_img = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        batch_img[0] = torch.from_numpy(norm_img)

        self.status = self.predict(batch_img)
        return self.status


class CoilOpenStatus(CoilStatusNet):
    def __init__(self, src_size, src_position):
        super(CoilOpenStatus, self).__init__(status_cnt=2, src_size=src_size, src_position=src_position, dst_size=128)

        self.add_conv_layer(channels=8, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=16, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=32, kernel_size=5, padding=2, pool_size=2)
        self.add_linear_layer_by_divider(divider=2)
        self.add_linear_layer_by_divider(divider=4)
        self.add_linear_layer(output_size=2)

        self.save_prefix = "开卷检测"


class CoilPosStatus(CoilStatusNet):
    def __init__(self, src_size, src_position):
        super(CoilPosStatus, self).__init__(status_cnt=2, src_size=src_size, src_position=src_position, dst_size=128)

        self.add_conv_layer(channels=8, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=16, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=32, kernel_size=5, padding=2, pool_size=2)
        self.add_linear_layer_by_divider(divider=2)
        self.add_linear_layer_by_divider(divider=4)
        self.add_linear_layer(output_size=2)

        self.save_prefix = "上卷检测"


class CoilExist(CoilStatusNet):
    def __init__(self, src_size, src_position):
        super(CoilExist, self).__init__(status_cnt=2, src_size=src_size, src_position=src_position, dst_size=128)

        self.add_conv_layer(channels=8, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=16, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=32, kernel_size=5, padding=2, pool_size=2)
        self.add_linear_layer_by_divider(divider=2)
        self.add_linear_layer_by_divider(divider=4)
        self.add_linear_layer(output_size=2)

        self.save_prefix = "带卷存在"


def main():
    mode = input('Network for coil open(O) or position(P) detection?')
    if mode == 'o' or mode == 'O':
        # specify the train and test set
        train_set = CoilImageSet("D:/01 自动上卷/开卷分类图片", class_cnt=2, input_size=1200, input_left_top=(0, 800),
                                 output_size=128)
    else:
        train_set = CoilImageSet("D:/01 自动上卷/上卷分类图片", class_cnt=2, input_size=1400, input_left_top=(0, 900),
                                 output_size=128)
    eval_set = train_set

    # check labeled images
    action = input('check labeled images? Yes(Y) or No(N)')
    if action == 'y' or action == 'Y':
        train_set.check_image_label()
        return

    # train the model
    if mode == 'o' or mode == 'O':
        Net = CoilOpenStatus(src_size=1200, src_position=(0, 800))
        Net.save_prefix = "开卷检测"
    else:
        Net = CoilPosStatus(src_size=1200, src_position=(0, 800))
        Net.save_prefix = "上卷检测"

    # train model
    Net.learn_rate = 0.001
    Net.l2_lambda = 0.002
    Net.dropout_rate = 0.4
    Net.save_dir = "D:/01 自动上卷/Python"
    Net.save_interval = 5000

    Net.init_model()
    Net.train(train_set, eval_set, epoch_cnt=5000, mini_batch=32)

    while True:
        Net.test_net(eval_set)
        if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
            break

if __name__ == '__main__':
    main()
