import cv2
import numpy as np
from enum import Enum


class ExposureStatus(Enum):
    NORMAL = 0   # 曝光正常
    OVER = 1     # 过曝
    UNDER = -1   # 欠曝


class Camera(object):
    def __init__(self):
        self.ROI = None
        self.image = None

    def __eval_clarity(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def __eval_exposure(self, image, over_exp_gray=240, under_exp_gray=20, over_under_thresh=0.1):
        """
        Evaluate if the camera is over- or under-exposure.
        :param image: the image
        :param over_exp_gray: the gray level of over-exposure
        :param under_exp_gray: the gray level of under-exposure
        :param over_under_thresh: the threshold to decide over- or under-exposure
        :return:
        """
        over_exp_percent = (image > over_exp_gray).mean()
        if over_exp_percent > over_under_thresh:
            return ExposureStatus.OVER

        under_exp_percent = (image < under_exp_gray).mean()
        if under_exp_percent > over_under_thresh:
            return ExposureStatus.UNDER

        return ExposureStatus.NORMAL

    def __get_ROI(self, mat, ROI=None):
        if ROI is None:
            return mat
        else:
            return mat[self.ROI[1]: self.ROI[3], self.ROI[0]: self.ROI[2]]

    def refresh(self):
        pass

    def get_image(self, width=None):
        return self.image

    def get_histogram(self, ROI=None):
        image = self.image if ROI is None else self.image[ROI[1]: ROI[1] + ROI[3], ROI[0]: ROI[0] + ROI[2]]
        hist, _ = np.histogram(image.flatten(), 256)
        dist = hist / hist.max()
        return hist, dist

    def draw_histogram(self, hist):
        img = np.zeros((100, 256), dtype=np.uint8)
        for i, h in enumerate(hist):
            v = int(h * 100)
            img[100 - v: 100, i] = 255
        return img

    def set_exposure_time(self, exp_time):
        pass

    def get_exposure_time(self):
        return 0

    def auto_exposure_time(self, min_exp_time, max_exp_time, adjust_cnt=4, ROIonly=False):
        exp_time = self.get_exposure_time()
        over_exp_time = max_exp_time
        under_exp_time = min_exp_time
        for _ in range(adjust_cnt):
            self.set_exposure_time(exp_time)
            image = self.get_image()
            if image is None:
                continue

            if ROIonly and self.ROI is not None:
                image = self.__get_ROI(image, self.ROI)

            exp_status = self.__eval_exposure(image)
            if exp_status == ExposureStatus.NORMAL:
                break
            elif exp_status == ExposureStatus.OVER:
                over_exp_time = exp_time
                exp_time = (under_exp_time + exp_time) / 2
            elif exp_status == ExposureStatus.UNDER:
                under_exp_time = exp_time
                exp_time = (over_exp_time + exp_time) / 2
               
        return exp_time

    def set_led(self, exp_time, flag):
        self.set_exposure_time(exp_time)
        image = self.get_image()
        
        exp_status = self.__eval_exposure(image)
        if exp_status == ExposureStatus.NORMAL:
            flag='NORMAL'
        elif exp_status == ExposureStatus.OVER:
            flag='OVER'
        elif exp_status == ExposureStatus.UNDER:
            flag='UNDER'
        return flag
    
    def gain_image(self, repeat_cnt,width=None, ROIonly=False):
        list_image=[]
        for _ in range(repeat_cnt):
            list_image.append(self.get_image(width))
        return list_image
                  
    def get_clearest_image1(self, list_image, repeat_cnt, width=None, ROIonly=False):
        clarity=[]
        for i in range(len(list_image)):
            clarity.append(self.__eval_clarity(image=list_image[i]))
        p=clarity.index(max(clarity))
            
        return list_image[p]
                  
    def get_clearest_image(self, repeat_cnt, width=None, ROIonly=False):
        best_image = self.get_image(width)
        if best_image is None:
            return None

        best_clarity = self.__eval_clarity(image=best_image)
        for _ in range(repeat_cnt):
            image = self.get_image(width)
            if image is None:
                continue

            if ROIonly and self.ROI is not None:
                image = self.__get_ROI(image, self.ROI)

            clarity = self.__eval_clarity(image)
            if clarity > best_clarity:
                best_image = image
                best_clarity = clarity
        return best_image
