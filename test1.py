import torch
import cv2
import torch.nn as nn
import numpy as np

class MyNet(nn.Module):
    def __init__(self, img_size, regress, conv_layers, linear_layers, dropout_rate):
        super(MyNet, self).__init__()
        self.img_size = img_size
        self.regress = regress

        self.conv_layers = torch.nn.Sequential()
        for i, layer in enumerate(conv_layers):
            in_channels = layer[0]
            out_channels = layer[1]
            kernel_size = layer[2]
            padding = layer[3]
            pool_size = layer[4]

            self.conv_layers.add_module('conv%d' % i,
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=1, padding=padding))
            self.conv_layers.add_module('conv_dropout%d' % i, nn.Dropout2d(dropout_rate))
            self.conv_layers.add_module('conv_act%d' % i, nn.ReLU())
            self.conv_layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=pool_size, stride=pool_size, padding=0))

        self.dense_layers = torch.nn.Sequential()
        for i in range(len(linear_layers) - 1):
            in_size = linear_layers[i][0]
            out_size = linear_layers[i][1]
            self.dense_layers.add_module('linear%d' % i, nn.Linear(in_size, out_size))
            self.dense_layers.add_module('dense_dropout%d' % i, nn.Dropout2d(dropout_rate))
            self.dense_layers.add_module('dense_act%d' % i, nn.ReLU())

        self.dense_layers.add_module('output', nn.Linear(linear_layers[-1][0], linear_layers[-1][1]))
        if not regress:
            self.dense_layers.add_module('LogSoftmax', nn.LogSoftmax(dim=1))

    def forward(self, input):
        out = input.view(-1, 1, self.img_size, self.img_size)
        out = self.conv_layers(out)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out

class CoilStatus():
    def __init__(self, src_size, src_position):
        self.status_cnt = 2
        self.src_size = src_size
        self.src_position = src_position

        self.regress = False
        self.img_size = 128
        self.img_channel = 1

        self.conv_layers = list()
        self.linear_layers = list()
        self.dropout_rate = 0.4
        self.NO_CUDA = False

        self.add_conv_layer(channels=8, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=16, kernel_size=5, padding=2, pool_size=2)
        self.add_conv_layer(channels=32, kernel_size=5, padding=2, pool_size=2)
        self.add_linear_layer_by_divider(divider=2)
        self.add_linear_layer_by_divider(divider=4)
        self.add_linear_layer(output_size=2)

    def analyze_one_frame(self, frame):
        img = frame[self.src_position[1]: self.src_position[1] + self.src_size,
                      self.src_position[0]: self.src_position[0] + self.src_size].copy()
        img = cv2.resize(img, (self.img_size, self.img_size))
        norm_img = self.normalize_image(img)

        batch_img = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        batch_img[0] = torch.from_numpy(norm_img)

        return self.predict(batch_img)

    def predict(self, batch_img):
        self.model.eval()
        batch_img = batch_img.to(self.device)
        outputs = self.model(batch_img)
        outputs = outputs.cpu().detach().numpy()
        return np.argmax(outputs[0])

    def init_model(self):
        self.device = self.try_gpu()
        self.model = MyNet(self.img_size, self.regress, self.conv_layers, self.linear_layers, self.dropout_rate)
        self.model = self.model.to(self.device)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        self.model = self.model.to(self.device)

    def try_gpu(self):
        """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
        device = torch.device('cpu') if self.NO_CUDA or not torch.cuda.is_available() else torch.device('cuda')
        #print("Using ", device)
        return device

    def normalize_image(self, image):
        mean = image.mean()
        std = image.std()
        norm_img = (image - mean) / std
        return norm_img

    def add_conv_layer(self, channels, kernel_size, padding, pool_size):
        if not self.conv_layers:
            in_channels = self.img_channel
        else:
            in_channels = self.conv_layers[-1][1]

        self.conv_layers.append((in_channels, channels, kernel_size, padding, pool_size))
        self.lst_conv_channels = channels

    def add_linear_layer(self, output_size):
        if not self.linear_layers:
            pool_size = 1
            for _, layer in enumerate(self.conv_layers):
                pool_size *= layer[4]

            pooled_img_size = int(self.img_size / pool_size)
            in_size = pooled_img_size * pooled_img_size * self.conv_layers[-1][1]
        else:
            in_size = self.linear_layers[-1][1]

        self.linear_layers.append((in_size, output_size))
        self.output_size = self.linear_layers[-1][1]

    def add_linear_layer_by_divider(self, divider):
        if not self.linear_layers:
            pool_size = 1
            for _, layer in enumerate(self.conv_layers):
                pool_size *= layer[4]

            pooled_img_size = int(self.img_size / pool_size)
            in_size = pooled_img_size * pooled_img_size * self.conv_layers[-1][1]
        else:
            in_size = self.linear_layers[-1][1]
        self.linear_layers.append((in_size, int(in_size / divider)))
        self.output_size = self.linear_layers[-1][1]

def arrayreset3(array):
    a = array[:, 0:len(array[0] - 2):3]#数组切片eg: b = a[2:7:2]   # 从索引 2 开始到索引 7 停止，间隔为 2
    b = array[:, 1:len(array[0] - 2):3]#这里abc分别是RGB三通道的像素值
    c = array[:, 2:len(array[0] - 2):3]
    a = a[:, :, None]
    b = b[:, :, None]
    c = c[:, :, None]
    m = np.concatenate((a, b, c), axis=2)#axis=2表示对应列的数组进行拼接
    return m

#初始化
def init():
    global load
    load = CoilStatus(src_size=1200, src_position=(0, 800))
    load.init_model()
    load.load_model("D:/autoload/Python/上卷检测2020-10-24-16-30.pt")

#接收图片并调用
def receive(image,mode):
    if (image.shape[1] - image.shape[0]) > image.shape[0]:
        frame1 = arrayreset3(image)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        frame1 = image
    global load
    pp = load.analyze_one_frame(frame1)
    return pp

if __name__ == '__main__':

    auto_loader = CoilStatus(src_size=1200, src_position=(0, 800))
    auto_loader.init_model()
    auto_loader.load_model("D:/autoload/Python/上卷检测2020-10-24-16-30.pt")

    frame=cv2.imread("D:/111.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    init()
    status = receive(frame,1)

    print("coil loaded") if status else print("coil not loaded")