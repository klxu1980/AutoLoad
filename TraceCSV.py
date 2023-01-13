import csv
import time


class TraceCSV(object):
    def __init__(self, file_name, open_write=True):
        if open_write:
            self.__init_open_write(file_name)
        else:
            self.__init_open_read(file_name)

        self.tracking_status = None
        self.plc_status = None

    def __del__(self):
        self.csv_file.close()

    def __init_open_write(self, file_name):
        self.csv_file = open(file_name + ".csv", "w", newline='')
        self.csv = csv.writer(self.csv_file)
        self.__write_caption()
        self.csv_start_time = time.perf_counter()

    def __init_open_read(self, file_name):
        self.csv_file = open(file_name, 'r')
        self.csv = csv.reader(self.csv_file)

    def __write_caption(self):
        row = []
        row.append("ellipse_exp, x")
        row.append("y")
        row.append("long")
        row.append("short")
        row.append("angle")

        row.append("ellipse_cnn, x")
        row.append("y")
        row.append("long")
        row.append("short")
        row.append("angle")

        row.append("ellipse_fit, x")
        row.append("y")
        row.append("long")
        row.append("short")
        row.append("angle")

        row.append("ellipse_exp_fit, x")
        row.append("y")
        row.append("long")
        row.append("short")
        row.append("angle")

        row.append("ellipse_vision, x")
        row.append("y")
        row.append("long")
        row.append("short")
        row.append("angle")

        row.append("ellipse_kalman, x")
        row.append("y")
        row.append("long")
        row.append("short")
        row.append("angle")

        row.append("coil_ellipse, x")
        row.append("y")
        row.append("long")
        row.append("short")
        row.append("angle")

        row.append("proc_interval")

        row.append("car_up")
        row.append("car_down")
        row.append("car_forward")
        row.append("car_backward")
        row.append("support_open")

        row.append("csv time")

        self.csv.writerow(row)

    def __write_ellipse(self, ellipse):
        if ellipse is None:
            ellipse = (0, 0, 0, 0, 0)
        elif len(ellipse) > 5:
            ellipse = ellipse[0:5]
        str = ["%1.3f" % value for value in ellipse]
        return str

    def write_file(self):
        row = []
        row += self.tracking_status
        row += self.plc_status
        row.append("%1.3f" % ((time.perf_counter() - self.csv_start_time) * 1000.0))
        self.csv.writerow(row)

    def record_tracking(self, tracker):
        row = []
        row += self.__write_ellipse(tracker.ellipse_exp)
        row += self.__write_ellipse(tracker.ellipse_cnn)
        row += self.__write_ellipse(tracker.ellipse_fit)
        row += self.__write_ellipse(tracker.ellipse_exp_fit)
        row += self.__write_ellipse(tracker.ellipse_vision)
        row += self.__write_ellipse(tracker.ellipse_kalman)
        row += self.__write_ellipse(tracker.coil_ellipse)
        row.append("%1.3f" % (tracker.proc_interval * 1000.0))
        self.tracking_status = row

    def record_plc(self, plc):
        row = []
        row.append("%d" % plc.car_up)
        row.append("%d" % plc.car_down)
        row.append("%d" % plc.car_forward)
        row.append("%d" % plc.car_backward)
        row.append("%d" % plc.support_open)
        self.plc_status = row

