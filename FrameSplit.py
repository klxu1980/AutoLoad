import cv2
import os
import glob
import time
import numpy as np


class VideoAnalyzer(object):
    def __init__(self, file_name, left, top, width, height):
        self.video = cv2.VideoCapture(file_name)
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.frame_cnt = int(self.video.get(7))
        self.cur_frame = 0

        self.edge_detector = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")

    def pre_process_image(self, image):
        image = np.float32(image) * (1.0 / 255.0)
        edge = self.edge_detector.detectEdges(image)
        return edge

    def get_frame(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
        _, frame = self.video.read()
        frame = frame[self.top: self.top + self.height, self.left: self.left + self.width, :]
        frame = self.pre_process_image(frame)
        return frame

    def forward(self):
        self.cur_frame = min(self.frame_cnt - 1, self.cur_frame + 1)

    def backward(self):
        self.cur_frame = max(0, self.cur_frame - 1)


def mark_video_frames(video_name, dst_dir):
    video = VideoAnalyzer(video_name, left=500, top=200, width=1500, height=1500)

    top_x = 500
    top_y = 500
    while True:
        frame = video.get_frame()
        image = frame.copy()

        cv2.line(image, (top_x - 100, top_y), (top_x + 100, top_y), color=(0, 255, 0))
        cv2.line(image, (top_x, top_y - 100), (top_x, top_y + 100), color=(0, 255, 0))
        cv2.imshow("video", image)

        key = cv2.waitKeyEx(10)
        if key == ord('s') or key == ord('S'):
            img_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ("-%d-%d.jpg" % (top_x, top_y))
            cv2.imwrite(dst_dir + img_name, frame)
            video.forward()
        elif key == ord(' '):
            video.forward()
        elif key == 2555904:
            top_x += 1
        elif key == 2424832:
            top_x -= 1
        elif key == 2490368:
            top_y -= 1
        elif key == 2621440:
            top_y += 1
        elif key == ord('e') or key == ord('E'):
            exit(0)

        video.forward()


def test_labeled_images(dir_name):
    os.chdir(dir_name)
    for file_name in glob.glob("*.jpg"):
        params = file_name.split('.')[0].split('-')
        top_x = int(params[6])
        top_y = int(params[7])
        image = cv2.imread(file_name)

        cv2.line(image, (top_x - 100, top_y), (top_x + 100, top_y), color=(0, 255, 0))
        cv2.line(image, (top_x, top_y - 100), (top_x, top_y + 100), color=(0, 255, 0))
        cv2.putText(image, file_name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("video", image)

        key = cv2.waitKeyEx(0)
        if key == ord('e') or key == ord('E'):
            exit(0)


if __name__ == '__main__':
   file_name = "D:\\自动上卷资料\\视频\\20200712161241836_20200713_092110_0 00_26_49-00_30_00.avi"  #"G:\\20220620\\14.20.25\\偏高.avi"
   mark_video_frames(video_name=file_name, dst_dir="G:\\20220620\\")
   test_labeled_images("G:\\20220721\\")

