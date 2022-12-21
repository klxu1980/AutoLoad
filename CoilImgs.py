import torch
import cv2
import random
import glob
import os

class CoilImageSet:
    def __init__(self, dir, class_cnt, input_size, input_left_top, output_size):
        self.img_size = output_size
        self.class_cnt = class_cnt

        self.images = list()
        for type in range(class_cnt):
            dir_class = dir + '/' + str(type)
            os.chdir(dir_class)
            for file_name in glob.glob("*.jpg"):
                img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)  # read image from the file
                img = img[input_left_top[1]: input_left_top[1] + input_size, input_left_top[0]: input_left_top[0] + input_size].copy()
                img = cv2.resize(img, (output_size, output_size))

                self.images.append([img, type])

        print(("%d images have been loaded from " % len(self.images)) + dir)

    def normalize_image(self, image):
        mean = image.mean()
        std = image.std()
        norm_img = (image - mean) / std
        return norm_img

    def random_sample(self, batch_size):
        batch_img = torch.zeros(batch_size, self.img_size, self.img_size, dtype=torch.float32)
        batch_label = torch.zeros(batch_size, dtype=torch.long)
        raw_images = list()

        random.seed()
        for i in range(batch_size):
            idx = random.randint(0, len(self.images) - 1)
            img = self.images[idx][0]
            norm_img = self.normalize_image(img)

            batch_img[i] = torch.from_numpy(norm_img)
            batch_label[i] = self.images[idx][1]
            raw_images.append(img)
        return batch_img, batch_label, raw_images

    def check_image_label(self):
        for i, img_lbl in enumerate(self.images):
            img = img_lbl[0]
            cv2.putText(img, "status = %d" % img_lbl[1], (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("", img)
            if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
                break

if __name__ == '__main__':
    img_set = CoilImageSet("D:/自动上卷/开卷分类图片", class_cnt=2, input_size=1200, input_top_left=(0, 800), output_size=128)
    img_set.check_image_label()