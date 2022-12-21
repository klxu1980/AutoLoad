import cv2
import os
import glob
import time


class VideoAnalyzer(object):
    def __init__(self, file_name, left, top, width, height):
        self.video = cv2.VideoCapture(file_name)
        self.left = left
        self.top = top
        self.width = width
        self.height = height

        self.frame_cnt = int(self.video.get(7))
        self.cur_frame = 0

    def get_frame(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
        _, frame = self.video.read()
        frame = frame[self.top: self.top + self.height, self.left: self.left + self.width, :]
        return frame

    def forward(self):
        self.cur_frame = min(self.frame_cnt - 1, self.cur_frame + 1)

    def backward(self):
        self.cur_frame = max(0, self.cur_frame - 1)