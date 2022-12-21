import cv2
import os
import glob
import time
import numpy as np


class VideoAnalyzer(object):
    def __init__(self, file_name, ROI=None):
        self.video = cv2.VideoCapture(file_name)
        self.ROI = ROI

        self.frame_cnt = int(self.video.get(7))
        self.cur_frame = self.frame_cnt - 1
        self.canny1 = 50
        self.canny2 = 150

    def get_frame(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
        _, frame = self.video.read()
        if self.ROI is not None:
            frame = frame[self.ROI[1]: self.ROI[3], self.ROI[0]: self.ROI[2], :]

        grad = frame
        grad = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
        grad = cv2.GaussianBlur(grad, (3, 3), 0)
        grad = cv2.Canny(grad, self.canny1, self.canny2)

        return frame, grad

    def forward(self):
        self.cur_frame = min(self.frame_cnt - 1, self.cur_frame + 1)

    def backward(self):
        self.cur_frame = max(0, self.cur_frame - 1)


if __name__ == '__main__':
    dir = "G:\\AutoAim\\20220620\\14.20.25\\"
    left = 1000
    top = 1500
    width = 400
    height = 500
    ROI = (left, top, left + width, top + height)

    file_name = dir + "正常.avi"
    video_low = VideoAnalyzer(dir + "偏低.avi", ROI)
    video_high = VideoAnalyzer(dir + "偏高.avi", ROI)
    video_ok = VideoAnalyzer(dir + "正常.avi", ROI)
    videos = (video_low, video_high, video_ok)

    frame, _ = video_low.get_frame()
    img_width = frame.shape[1]
    img_height = frame.shape[0]

    cur_video = 0
    while True:
        screen = np.zeros((img_height, img_width * 4, 3), dtype=np.uint8)
        img_grad = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        for i in range(3):
            frame, grad = videos[i].get_frame()
            img_grad[:, :, i] = grad
            screen[:, (i + 1) * img_width: (i + 2) * img_width, :] = frame
        screen[:, 0: img_width, :] = img_grad

        cv2.putText(screen, "Blue = low, frame = %d" % videos[0].cur_frame, (20, height - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(screen, "Green = high, frame = %d" % videos[1].cur_frame, (20, height - 100 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(screen, "Red = ok, frame = %d" % videos[2].cur_frame, (20, height - 100 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(screen, "current channel = %d" % cur_video, (20, height - 100 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("video", screen)

        key = cv2.waitKeyEx(0)
        if key == ord('s') or key == ord('S'):
            img_name = time.strftime("%Y_%m_%d_%H_%M_%S.bmp", time.localtime())
            cv2.imwrite("d:\\AutoLoadImages\\" + img_name, screen)
        elif key == 2555904:
            videos[cur_video].forward()
        elif key == 2424832:
            videos[cur_video].backward()
        elif key == 2490368:
            cur_video = (cur_video + 1) % 3
        elif key == ord('1'):
            for i in range(3):
                videos[i].canny1 += 1
        elif key == ord('2'):
            for i in range(3):
                videos[i].canny1 -= 1
        elif key == ord('3'):
            for i in range(3):
                videos[i].canny2 += 1
        elif key == ord('4'):
            for i in range(3):
                videos[i].canny2 -= 1
        elif key == ord('e') or key == ord('E'):
            exit(0)
