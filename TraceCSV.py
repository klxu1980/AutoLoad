import csv
import time


class TraceCSV(object):
    def __init__(self, file_name):
        csv_file = open(file_name + ".csv", "w", newline='')
        self.csv = csv.writer(csv_file)
        self.csv_start_time = time.perf_counter()

        self.tracking_status = None
        self.plc_status = None

        self.ellipse = (0, 0, 0, 0, 0)
        self.proc_interval = 0.0

    def __write_ellipse(self, ellipse):
        if ellipse is None:
            ellipse = (0, 0, 0, 0, 0)
        str = ["%1.3f" % value for value in ellipse]
        return str

    def write_file(self):
        row = []
        row += self.tracking_status
        row += self.plc_status
        row.append(("%1.2f" % (time.perf_counter() - self.csv_start_time)))
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
        row.append("%1.3f" % tracker.proc_interval)
        self.tracking_status = row

    def record_plc(self, plc):
        row = []
        row.append("%d" % plc.car_up)
        row.append("%d" % plc.car_down)
        row.append("%d" % plc.car_forward)
        row.append("%d" % plc.car_backward)
        row.append("%d" % plc.support_open)
        self.plc_status = row

