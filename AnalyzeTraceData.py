import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


# 注意：该文件已不使用，带卷轨迹的分析处理在AnalyzeTrace.py中完成


def read_trace_stat(file_name):
    """
    从一个轨迹文件中读取带卷的中心位置，以及
    :param file_name:
    :return:
    """
    trace_stat = {}
    csv_reader = csv.reader(open(file_name, 'r'))
    for line in csv_reader:
        center = (float(line[0]), float(line[1]))   # 椭圆中心位置
        values = [float(line[2]), float(line[3]), float(line[4]), float(line[5])]   # 椭圆数量、长轴、短轴、角度平均值
        trace_stat[center] = values
    return trace_stat


def write_trace_stat(file_name, trace_stat):
    csv_writer = csv.writer(open(file_name, 'w', newline=''))
    for row in trace_stat.items():
        line = (str(row[0][0]), str(row[0][1]),
                str(row[1][0]),
                str(row[1][1]), str(row[1][2]), str(row[1][3]),
                str(row[1][4]), str(row[1][5]), str(row[1][6])
                )
        csv_writer.writerow(line)


def read_trace_csv_raw(file_name, raw_list):
    """
    从一个带卷轨迹csv文件中，读取带卷轨迹原始信息(x, y, long, short, angle)，存储到raw_list中
    :param file_name:
    :param raw_list:
    :return:
    """
    csv_reader = csv.reader(open(file_name, 'r'))
    for line in csv_reader:
        ellipse = np.array((int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])))
        raw_list.append(ellipse)


def save_all_csv(dir, dst_file):
    csv_writer = csv.writer(open(dst_file, 'w', newline=''))

    file_cnt = 0
    for root, dirs, files in os.walk(dir):
        for _, file in enumerate(files):
            if os.path.splitext(file)[-1] == ".csv" and ".avi" in os.path.splitext(file)[0]:
                raw_list = list()
                file_name = root + "\\" + file
                read_trace_csv_raw(file_name, raw_list)

                file_cnt += 1
                csv_writer.writerow((str(file_cnt), file_name))
                for _, row in enumerate(raw_list):
                    line = (str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]))
                    csv_writer.writerow(line)


def calc_trace_avg_std(raw_list):
    """
    统计计算每一个轨迹点上带卷内椭圆的参数(长轴，短轴，角度)平均值
    :param raw_list:
    :return:
    """
    # 统计字典
    trace_stat = {}

    # 计算椭圆参数平均值
    for _, ellipse in enumerate(raw_list):
        # 误差超过范围要求的椭圆被剔除
        if ellipse[0] < 0:
            continue

        center = (ellipse[0], ellipse[1])
        if center not in trace_stat:
            trace_stat[center] = np.zeros(7, dtype=np.float32)
        stat = trace_stat[center]

        stat[0] += 1                 # 样本数量
        stat[1:4] += ellipse[2:5]    # 累积长轴、短轴、角度
    for row in trace_stat.items():
        row[1][1:4] /= row[1][0]

    # 计算椭圆参数标准差
    for _, ellipse in enumerate(raw_list):
        # 误差超过范围要求的椭圆被剔除
        if ellipse[0] < 0:
            continue

        stat = trace_stat[(ellipse[0], ellipse[1])]
        stat[4:7] += (ellipse[2:5] - stat[1:4])**2

    for row in trace_stat.items():
        if row[1][0] > 1:
            row[1][4:7] /= row[1][0] - 1
        row[1][4:7] = np.sqrt(row[1][4:7])

    return trace_stat


def remove_gross_error(raw_list, trace_stat):
    for _, ellipse in enumerate(raw_list):
        if ellipse[0] < 0:
            continue

        stat = trace_stat[(ellipse[0], ellipse[1])]
        if stat[0] > 1:
            err = np.fabs(ellipse[2:5] - stat[1:4])
            if err[0] > stat[4] or err[1] > stat[5] or err[2] > stat[6]:
                ellipse[0] = -1


def stat_ellipse_params(dir):
    """
    统计分析带卷轨迹数据
    :param dir:
    :return:
    """
    # AutoLoad能够对每一个带卷视频进行轨迹跟踪，将每一帧中的带卷轨迹(以及内椭圆信息)保存到csv文件中
    # 首先枚举所有的带卷轨迹数据文件，将原始数据写入raw_list中
    raw_list = list()
    for root, dirs, files in os.walk(dir):
        for _, file in enumerate(files):
            if os.path.splitext(file)[-1] == ".csv" and ".avi" in os.path.splitext(file)[0]:
                file_name = root + "\\" + file
                read_trace_csv_raw(file_name, raw_list)

    # 计算椭圆参数平均值和标准差
    trace_stat = calc_trace_avg_std(raw_list)
    remove_gross_error(raw_list, trace_stat)
    trace_stat = calc_trace_avg_std(raw_list)

    # 保存统计结果
    write_trace_stat(dir + "\\trace_stat.csv", trace_stat)


def stat_aiming_trace(dir):
    raw_list = list()
    for root, dirs, files in os.walk(dir):
        for _, file in enumerate(files):
            if os.path.splitext(file)[-1] == ".csv" and ".avi" in os.path.splitext(file)[0]:
                file_name = root + "\\" + file
                read_trace_csv_raw(file_name, raw_list)

    # 统计轨迹每个x的平均值(累加值)、最大值和最小值
    trace_stat = {}
    for _, ellipse in enumerate(raw_list):
        x = ellipse[0]
        y = ellipse[1]
        if x not in trace_stat:
            trace_stat[x] = [1.0, y, y, y]
        else:
            trace_stat[x][0] += 1
            trace_stat[x][1] += y
            trace_stat[x][2] = min(trace_stat[x][2], y)
            trace_stat[x][3] = max(trace_stat[x][2], y)

    # 线性拟合轨迹
    X = []
    AvgY = []
    MinY = []
    MaxY = []

    # 计算平均值
    for row in trace_stat.items():
        X.append(row[0])
        AvgY.append(row[1][1] / row[1][0])
        MinY.append(row[1][2])
        MaxY.append(row[1][3])
    trace_avg = np.polyfit(X, AvgY, 1)
    trace_min = np.polyfit(X, MinY, 1)
    trace_max = np.polyfit(X, MaxY, 1)

    # 保存
    csv_writer = csv.writer(open(dir + "\\trace_stat.csv", 'w', newline=''))
    line = (str(trace_avg[0]), str(trace_avg[1]))
    csv_writer.writerow(line)
    line = (str(trace_min[0]), str(trace_min[1]))
    csv_writer.writerow(line)
    line = (str(trace_max[0]), str(trace_max[1]))
    csv_writer.writerow(line)

    print(trace_avg)
    print(trace_min)
    print(trace_max)


def show_trace_stat(file_name):
    trace_map = np.zeros((2000, 1500), dtype=np.int32)

    trace_stat = read_trace_stat(file_name)
    for row in trace_stat.items():
        cnt = int(row[1][0])
        if cnt > 1:
            x = int(row[0][0]) - 1000
            y = 2000 - int(row[0][1])
            if 0 <= x < trace_map.shape[1] and 0 <= y < trace_map.shape[0]:
                trace_map[y][x] = min(cnt, 50)

    grid_x = np.arange(0, 1500)
    grid_y = np.arange(0, 2000)
    grid = np.array(np.meshgrid(grid_x, grid_y))

    plt.contourf(grid[0], grid[1], trace_map, 100, cmap='RdBu_r')
    plt.colorbar()
    plt.title("Coil trace")
    plt.show()


if __name__ == '__main__':
    save_all_csv("E:\\20220521-20220621数据\\视频数据", "e:\\历史数据.csv")
    #stat_ellipse_params("E:\\20220521-20220621数据\\视频数据")
    #show_trace_stat("E:\\20220521-20220621数据\\视频数据\\trace_stat.csv")
    #stat_aiming_trace("E:\\20220521-20220621数据\\视频数据")