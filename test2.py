
import cv2
import numpy as np

from test1 import CoilStatus

class openstatus():
    def __init__(self):
        self.coil_pos_status = CoilStatus(src_size=1200, src_position=(0, 800))
        self.coil_pos_status.init_model()
        self.coil_pos_status.load_model("D:/autoload/Python/上卷检测2020-10-24-16-30.pt")
        self.loading_stage = 0
        self.saving_video = False
        self.keytime = 10
    def run_one_img(self, frame):
        # 计算获取中心点位置(卡尔曼滤波后的位置，卷积网络计算位置)
        status = self.coil_pos_status.analyze_one_frame(frame)
        return status

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
def init_class():
    print("ok")
    global load
    load = openstatus()
    return 1
#接收图片并调用
def receive(image,mode):
    if (image.shape[1] - image.shape[0]) > image.shape[0]:
        frame = arrayreset3(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame = image
    global load
    pp = load.run_one_img(frame)
    return pp

if __name__ == '__main__':

    auto_loader = openstatus()

    frame=cv2.imread("D:/111.jpg")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    status = auto_loader.run_one_img(frame)

    print("coil loaded") if status else print("coil not loaded")