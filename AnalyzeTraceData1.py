import csv
import glob
import numpy as np
import matplotlib.pyplot as plt


def read_trace_stat(file_name):
    trace_stat = {}
    csv_reader = csv.reader(open(file_name, 'r'))
    for line in csv_reader:
        center = (float(line[0]), float(line[1]))   # 椭圆中心位置
        if 0 <= center[0] < 4000 and 0 <= center[1] < 4000:
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
    csv_reader = csv.reader(open(file_name, 'r'))
    for line in csv_reader:
        if 0 <= int(line[0]) < 4000 and 0 <= int(line[1]) < 4000:
          ellipse = np.array((int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])))
          raw_list.append(ellipse)


def analyze_trace_set_file(file_name):
    # read trace data from csv file
    csv_reader = csv.reader(open(file_name, 'r'))

    # 轨迹数据来源于不同文件，将每个文件中同一点的椭圆数据进行平均，写入trace_data中
    trace_data = []
    raw_one_file = []
    for line in csv_reader:
        if len(line[1]) > 10:
            trace_stat = calc_trace_avg_std(raw_one_file)
            for row in trace_stat.items():
                trace_data.append(np.array((row[0][0], row[0][1], row[1][1], row[1][2], row[1][3])))
            raw_one_file = []
        elif 0 <= int(line[0]) < 4000 and 0 <= int(line[1]) < 4000:
            ellipse = np.array((int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])))
            raw_one_file.append(ellipse)

    # 将每一个位置的椭圆归类到一个list中
    point_set = {}
    for _, row in enumerate(trace_data):
        center = (trace_data[0], trace_data[1])
        if center not in point_set:
            point_set[center] = []
        else:
            point_set[center].append(trace_data)

    # 对每个位置的椭圆数据进行统计分析
    for ellipse_list in point_set:


def read_all_trace_from_csv(file_name):
    traces = []

    csv_reader = csv.reader(open(file_name, 'r'))
    raw_list = []
    for line in csv_reader:
        if len(line[1]) > 10:
            trace_stat = calc_trace_avg_std(raw_list)
            for row in trace_stat.items():
                traces.append(np.array((row[0][0], row[0][1], row[1][1], row[1][2], row[1][3])))
            raw_list = []
        elif 0 <= int(line[0]) < 4000 and 0 <= int(line[1]) < 4000:
            ellipse = np.array((int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])))
            raw_list.append(ellipse)

    # 计算椭圆参数平均值和标准差
    trace_stat = calc_trace_avg_std(traces)
    remove_gross_error(traces, trace_stat)
    trace_stat = calc_trace_avg_std(traces)

    # 保存统计结果
    write_trace_stat("G:\\trace_stat.csv", trace_stat)

    return traces


def calc_trace_avg_std(raw_list):
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

        stat[0] += 1
        stat[1:4] += ellipse[2:5]
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
    raw_list = list()
    for file_name in glob.glob(dir + "\\*.csv"):
        if "trace_stat" not in file_name:
            read_trace_csv_raw(file_name, raw_list)

    # 计算椭圆参数平均值和标准差
    trace_stat = calc_trace_avg_std(raw_list)
    remove_gross_error(raw_list, trace_stat)
    trace_stat = calc_trace_avg_std(raw_list)

    # 保存统计结果
    write_trace_stat(dir + "\\trace_stat.csv", trace_stat)


def stat_aiming_trace(dir):
    raw_list = list()
    for file_name in glob.glob(dir + "\\*.csv"):
        if "trace_stat" not in file_name:
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
    cnt_stat = {}

    trace_stat = read_trace_stat(file_name)
    for row in trace_stat.items():
        cnt = int(row[1][0])
        if cnt not in cnt_stat:
            cnt_stat[cnt] = 0
        else:
            cnt_stat[cnt] += 1

        if cnt > 1:
            x = int(row[0][0]) - 1000
            y = 2000 - int(row[0][1])
            if 0 <= x < trace_map.shape[1] and 0 <= y < trace_map.shape[0]:
                trace_map[y][x] = min(cnt, 50)
    print(sorted(cnt_stat.items()))

    grid_x = np.arange(0, 1500)
    grid_y = np.arange(0, 2000)
    grid = np.array(np.meshgrid(grid_x, grid_y))

    plt.contourf(grid[0], grid[1], trace_map, 100, cmap='RdBu_r')
    plt.colorbar()
    plt.title("Coil trace")
    plt.show()


if __name__ == '__main__':
    read_all_trace_from_csv("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\历史数据.csv")
    show_trace_stat("G:\\trace_stat.csv")
    #stat_ellipse_params("E:\\01 我的设计\\05 智通项目\\04 自动上卷")
    #show_trace_stat("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\trace_stat.csv")
    #stat_aiming_trace("G:\\01 自动上卷视频\\对中视频\\正常")