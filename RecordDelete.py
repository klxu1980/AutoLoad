import os
import glob


def delete_small_record(dir, size):
    file_list = []
    for root, dirs, files in os.walk(dir):
        for _, file in enumerate(files):
            if os.path.splitext(file)[-1] == ".mp4" and "processed" in os.path.splitext(file)[0]:
                file_path = root + "\\" + file
                if os.path.getsize(file_path) < size:
                    file_time = file.split('-')[0]
                    file_list.append(root + "\\" + file_time)

    for file in file_list:
        os.remove(file + "-original.mp4")
        os.remove(file + "-processed.mp4")
        os.remove(file + ".csv")


if __name__ == '__main__':
    delete_small_record("E:\\NewTest", size=10000*1024)