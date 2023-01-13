import csv
import os
import numpy as np


def read_trace_stat(file_name):
    """
    从一个轨迹文件中读取带卷的中心位置，以及
    :param file_name:
    :return:
    """
    csv_reader = csv.reader(open(file_name, 'r'))

    line_idx = 0
    lst_x = 0
    lst_y = 0

    stat_up = []
    stat_down = []
    stat_forward_h = []
    stat_backward_h = []
    stat_forward_l = []
    stat_backward_l = []
    for line in csv_reader:
        if line_idx < 2:
            if line_idx > 0:
                lst_x = float(line[25])
                lst_y = float(line[26])
        else:
            delta_x = float(line[25]) - lst_x
            delta_y = float(line[26]) - lst_y
            lst_x = float(line[25])
            lst_y = float(line[26])

            interval = float(line[35]) * 0.001
            if interval < 0.1:
                continue
            values = (delta_x / interval, delta_y / interval)

            car_up = int(line[36])
            car_down = int(line[37])
            car_forward = int(line[38])
            car_backward = int(line[39])

            if car_up:
                stat_up.append(values)
            elif car_down:
                stat_down.append(values)
            elif car_forward:
                if lst_y < 1300:
                    stat_forward_h.append(values)
                else:
                    stat_forward_l.append(values)
            elif car_backward:
                if lst_y < 1300:
                    stat_backward_h.append(values)
                else:
                    stat_backward_l.append(values)
        line_idx += 1
    return stat_up, stat_down, stat_forward_h, stat_backward_h, stat_forward_l, stat_backward_l


def analyze_all_csv(dir):
    stat_up = []
    stat_down = []
    stat_forward_h = []
    stat_backward_h = []
    stat_forward_l = []
    stat_backward_l = []
    for root, dirs, files in os.walk(dir):
        for _, file in enumerate(files):
            if os.path.splitext(file)[-1] == ".csv":
                file_name = root + "\\" + file
                up, down, forward_h, backward_h, forward_l, backward_l = read_trace_stat(file_name)
                stat_up += up
                stat_down += down
                stat_forward_h += forward_h
                stat_backward_h += backward_h
                stat_forward_l += forward_l
                stat_backward_l += backward_l

    stat_up = np.array(stat_up)
    stat_down = np.array(stat_down)
    stat_forward_h = np.array(stat_forward_h)
    stat_backward_h = np.array(stat_backward_h)
    stat_forward_l = np.array(stat_forward_l)
    stat_backward_l = np.array(stat_backward_l)

    return np.mean(stat_up, axis=0), np.mean(stat_down, axis=0), np.mean(stat_forward_h, axis=0), \
           np.mean(stat_backward_h, axis=0), np.mean(stat_forward_l, axis=0), np.mean(stat_backward_l, axis=0)


if __name__ == '__main__':
    up, down, forward_h, backward_h, forward_l, backward_l = analyze_all_csv("E:\\NewTest")
    #print("上升 = %1.3f, 下降 = %1.3f, 前进(高) = %1.3f, 后退(高) = %1.3f, 前进(低) = %1.3f, 后退(低) = %1.3f" % (up, down, forward_h, backward_h, forward_l, backward_l))
    print("上升", up)
    print("下降", down)
    print("前进(高)", forward_h)
    print("后退(高)", backward_h)
    print("前进(低)", forward_l)
    print("后退(低)", backward_l)