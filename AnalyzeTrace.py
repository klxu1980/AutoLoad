import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


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
    """
    AutoLoad可以对每一个视频文件进行处理，将带卷轨迹和内椭圆信息存储在一个csv文件中
    save_all_csv()进一步将所有这些csv文件中的带卷信息存储到一个csv文件中，以便于分析处理带卷轨迹
    :param dir:
    :param dst_file:
    :return:
    """
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
    计算轨迹中椭圆参数的平均值和标准差
    :param raw_list: 椭圆轨迹参数列表，每个元素是一次椭圆检测结果，包含(x坐标，y坐标，长轴，短轴，角度)
    :return: 以列表形式输出每个(x, y)坐标上椭圆参数的平均值。
             每个列表元素为(x坐标，y坐标，长轴均值，短轴均值，角度均值，长轴标准差，短轴标准差，角度标准差，样本数量)
    """
    # 统计字典
    trace_stat = {}

    # 计算椭圆参数平均值
    for _, ellipse in enumerate(raw_list):
        center = (ellipse[0], ellipse[1])
        if center not in trace_stat:
            trace_stat[center] = np.zeros(4, dtype=np.float32)
        stat = trace_stat[center]

        stat[0] += 1
        stat[1:4] += ellipse[2:5]
    for row in trace_stat.items():
        row[1][1:4] /= row[1][0]

    """
    # 计算椭圆参数误差平方和
    for _, ellipse in enumerate(raw_list):
        stat = trace_stat[(ellipse[0], ellipse[1])]
        stat[4:7] += (ellipse[2:5] - stat[1:4])**2

    # 计算椭圆参数标准差
    avg_trace = []
    for row in trace_stat.items():
        if row[1][0] > 1:
            row[1][4:7] /= row[1][0] - 1
        row[1][4:7] = np.sqrt(row[1][4:7])

        avg_trace.append(np.array((
            row[0][0], row[0][1],              # 位置坐标
            row[1][1], row[1][2], row[1][3],   # 参数平均值
            row[1][4], row[1][5], row[1][6],   # 参数标准差
            row[1][0]                          # 样本数量
        )))
    """
    avg_trace = []
    for row in trace_stat.items():
        avg_trace.append(np.array((
            row[0][0], row[0][1],  # 位置坐标
            row[1][1], row[1][2], row[1][3]
        )))

    return avg_trace


def read_trace_csv_raw(file_name):
    raw_list = []
    csv_reader = csv.reader(open(file_name, 'r'))
    for line in csv_reader:
        if 0 <= int(line[0]) < 5000 and 0 <= int(line[1]) < 5000:
          ellipse = np.array((int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])))
          raw_list.append(ellipse)
    return raw_list


def read_trace_from_one_csv(file_name):
    raw_list = read_trace_csv_raw(file_name)
    avg_list = calc_trace_avg_std(raw_list)
    return avg_list


def read_traces_from_dir(root_dir):
    traces = []
    for root, dirs, files in os.walk(root_dir):
        for _, file in enumerate(files):
            if os.path.splitext(file)[-1] == ".csv":
                file_path = root + "\\" + file
                trace = read_trace_from_one_csv(file_path)
                traces += trace
    return traces


def read_all_traces_from_one_csv(file_name):
    all_traces = []
    csv_reader = csv.reader(open(file_name, 'r'))

    raw_list = []
    for line in csv_reader:
        if len(line[1]) > 10:
            trace = calc_trace_avg_std(raw_list)
            all_traces += trace
            raw_list = []
        elif 0 <= int(line[0]) < 5000 and 0 <= int(line[1]) < 5000:
            ellipse = np.array((int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])))
            raw_list.append(ellipse)
    return all_traces


def calc_ellipse_avg_std(samples):
    avg = np.zeros((1, 3), dtype=np.float32)
    std = np.zeros((1, 3), dtype=np.float32)

    for _, ellipse in enumerate(samples):
        avg += ellipse[2:5]
    avg /= len(samples)

    for _, ellipse in enumerate(samples):
        std += (ellipse[2:5] - avg)**2
    std /= (len(samples) - 1)
    std = np.sqrt(std)

    return avg, std


def remove_gross_error(samples, min_sample_cnt, max_std):
    while len(samples) >= min_sample_cnt:
        s = np.array(samples)
        avg = np.average(s, axis=0)
        std = np.std(s, axis=0)
        if all(std[2:5] < max_std):
            return np.concatenate((avg, std[2:5]))

        err = np.fabs(s - avg)[:, 2:5] / max_std
        err = np.sum(err**2, axis=1)
        max_idx = np.argmax(err)
        del samples[max_idx]

    return None


def clear_gross_error(traces, min_sample_cnt, max_std):
    samples = {}
    for _, ellipse in enumerate(traces):
        center = (ellipse[0], ellipse[1])
        if center not in samples:
            samples[center] = []
        samples[center].append(ellipse)

    trace_stat = []
    for row in samples.items():
        avg_std = remove_gross_error(samples=row[1], min_sample_cnt=min_sample_cnt, max_std=max_std)
        if avg_std is not None:
            trace_stat.append(avg_std)
    return trace_stat


def write_trace_stat(file_name, trace_stat):
    csv_writer = csv.writer(open(file_name, 'w', newline=''))
    for _, ellipse in enumerate(trace_stat):
        line = (str(ellipse[0]), str(ellipse[1]),
                str(ellipse[2]), str(ellipse[3]), str(ellipse[4]),
                str(ellipse[5]), str(ellipse[6]), str(ellipse[7]))
        csv_writer.writerow(line)


def read_trace_stat(file_name):
    trace_stat = {}
    csv_reader = csv.reader(open(file_name, 'r'))
    for line in csv_reader:
        center = (float(line[0]), float(line[1]))   # 椭圆中心位置
        values = [float(line[2]), float(line[3]), float(line[4])]   # 长轴、短轴、角度平均值
        trace_stat[center] = values
    return trace_stat


def show_trace_stat(file_name):
    trace_map = np.zeros((2000, 1500), dtype=np.int32)
    trace_stat = read_trace_stat(file_name)
    for row in trace_stat.items():
        x = int(row[0][0]) - 1000
        y = 2000 - int(row[0][1])
        if 0 <= x < trace_map.shape[1] and 0 <= y < trace_map.shape[0]:
            trace_map[y][x] = 1

    grid_x = np.arange(0, 1500)
    grid_y = np.arange(0, 2000)
    grid = np.array(np.meshgrid(grid_x, grid_y))

    plt.contourf(grid[0], grid[1], trace_map, 100, cmap='RdBu_r')
    plt.colorbar()
    plt.title("Coil trace")
    plt.show()


if __name__ == '__main__':
    # 如果需要把所有轨迹文件转存到一个csv文件中，则使用该代码
    #save_all_csv("E:\\20220521-20220621数据\\视频数据", "e:\\历史数据.csv")

    traces = read_all_traces_from_one_csv("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\历史数据.csv")

    max_std = np.array((5.0, 5.0, 5.0))
    trace_stat = clear_gross_error(traces=traces, min_sample_cnt=5, max_std=max_std)

    write_trace_stat("G:\\trace_stat.csv", trace_stat)
    show_trace_stat("G:\\trace_stat.csv")
