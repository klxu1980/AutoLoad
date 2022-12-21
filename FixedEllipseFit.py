import cv2
import numpy as np


def get_ellipse(ellipse_temp):
    canvas = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(canvas, (256, 256), (ellipse_temp[0], ellipse_temp[1]),
                ellipse_temp[2], ellipse_temp[3], ellipse_temp[4], 255,
                ellipse_temp[5])
    ellipse = np.array(np.where(canvas > 0)) - 256
    return ellipse


def get_ellipse_cluster(ellipse, angle_begin, angle_end, width, offset):
    ellipse_idx = get_ellipse(ellipse, angle_begin, angle_end, width)
    ellipse_cluster = np.zeros((2, offset**2, ellipse_idx.shape[1]), dtype=np.int32)


def fig_ellipse(frame, ellipse_temp, init_center, offset):
    ellipse = get_ellipse(ellipse_temp)

    max_fitness = 0
    bst_center = None
    for x in range(-int(offset / 2), int(offset / 2) + 1):
        for y in range(-int(offset / 2), int(offset / 2) + 1):
            cx = init_center[0] + x
            cy = init_center[1] + y
            fitness = np.sum(frame[ellipse[0] + cx, ellipse[1] + cy]]**2)
            if fitness > max_fitness:
                max_fitness = fitness
                bst_center = (cx, cy)

    return bst_center



"""
canvas = np.zeros((512, 512), dtype=np.uint8)
ellipse = get_ellipse((0, 0, 100, 50, 30), 0, 270, 3) + 256
canvas[ellipse[0], ellipse[1]] = 255
cv2.imshow("", canvas)
cv2.waitKey(0)
"""