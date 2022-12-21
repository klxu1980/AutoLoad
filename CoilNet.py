import cv2
import numpy as np
import datetime
from BaseImageNet import BaseImageNet
from ImageSet import ImageSet
from ImageSet import LabelType


class CoilNet(BaseImageNet):
    def __init__(self, img_size, output_size):
        super(CoilNet, self).__init__(regress=True, img_size=img_size, img_channels=1)

    def label2pixel(self, label):
        return int((label + 1.0) * self.model.img_size / 2)

    def label2angle(self, label):
        return int((label + 1.0) * 90)

    def predict(self, batch_img):
        outputs = super().predict(batch_img)

        params = list()
        for row in range(outputs.shape[0]):
            values = np.zeros(self.output_size, np.float32)
            for col in range(self.output_size):
                # the output may contain 2 or 5 elements.
                # when only center is required, it has 2 elements for center coordination;
                # when the whole ellipse is required, it has 5 elements, center, size and angle
                if col % 5 < 4:
                    values[col] = self.label2pixel(outputs[row][col])
                else:
                    values[col] = self.label2angle(outputs[row][col])
            params.append(values)
        return params

    def train(self, train_set, eval_set, epoch_cnt, mini_batch, loss_func=None):
        super().train(train_set, eval_set, epoch_cnt, mini_batch, loss_func)

    def test_net(self, eval_set):
        while True:
            batch_img, batch_label, raw_img = eval_set.random_sample(32)
            params = self.predict(batch_img)

            for i in range(len(params)):
                img1 = cv2.cvtColor(raw_img[i], cv2.COLOR_GRAY2BGR)
                img2 = img1.copy()

                cv2.circle(img1, (self.label2pixel(batch_label[i][0]), self.label2pixel(batch_label[i][1])), 2,
                           (0, 0, 255), 2)
                if self.output_size > LabelType.CenterOnly:
                    cv2.ellipse(img1, (self.label2pixel(batch_label[i][0]), self.label2pixel(batch_label[i][1])),
                                (self.label2pixel(batch_label[i][2]), self.label2pixel(batch_label[i][3])),
                                self.label2angle(batch_label[i][4]), 0, 360, (0, 0, 255), 1)

                cv2.circle(img1, (int(params[i][0] + 0.5), int(params[i][1] + 0.5)), 2, (0, 255, 0), 2)
                if self.output_size > LabelType.CenterOnly:
                    cv2.ellipse(img1, (int(params[i][0] + 0.5), int(params[i][1] + 0.5)),
                                (int(params[i][2]), int(params[i][3])), int(params[i][4]), 0, 360,
                                (0, 255, 0), 1)

                img = np.hstack((img1, img2))
                cv2.imshow("", img)

                key = cv2.waitKey(0)
                if key == ord('q'):
                    return
                elif key == ord('s'):
                    img = np.hstack((img1, img2, img2))
                    cv2.imwrite("G:\\CoilAnaly\\" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.jpg"), img)

    def denoise_test(self, eval_set):
        while True:
            batch_img, batch_label, raw_img = eval_set.random_sample(32)
            params = self.predict(batch_img)

            for i in range(len(params)):
                img1 = raw_img[i]
                img2 = img1.copy()

                mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
                cv2.ellipse(mask, (int(params[i][0] + 0.5), int(params[i][1] + 0.5)),
                                (int(params[i][2]), int(params[i][3])), int(params[i][4]), 0, 360,
                                255, 32)
                mask = cv2.blur(mask, (32, 32), 0)
                mask = mask / 255.0
                img1 = (img1 * mask).astype(np.uint8)

                """
                cv2.circle(img1, (self.label2pixel(batch_label[i][0]), self.label2pixel(batch_label[i][1])), 2,
                           (0, 0, 255), 2)
                if self.output_size > LabelType.CenterOnly:
                    cv2.ellipse(img1, (self.label2pixel(batch_label[i][0]), self.label2pixel(batch_label[i][1])),
                                (self.label2pixel(batch_label[i][2]), self.label2pixel(batch_label[i][3])),
                                self.label2angle(batch_label[i][4]), 0, 360, (0, 0, 255), 1)

                cv2.circle(img1, (int(params[i][0] + 0.5), int(params[i][1] + 0.5)), 2, (0, 255, 0), 2)
                if self.output_size > LabelType.CenterOnly:
                    cv2.ellipse(img1, (int(params[i][0] + 0.5), int(params[i][1] + 0.5)),
                                (int(params[i][2]), int(params[i][3])), int(params[i][4]), 0, 360,
                                (0, 255, 0), 1)
                """

                img = np.hstack((img1, img2))
                cv2.imshow("", img)

                key = cv2.waitKey(0)
                if key == ord('q'):
                    return
                elif key == ord('s'):
                    img = np.hstack((img1, img2, img2))
                    cv2.imwrite("G:\\CoilAnaly\\" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.jpg"), img)


def main():
    # the main is used for training and testing the neural network
    # there are basically following parameters that need to be specified
    # 1. size of the image, normally 128, 256, 512
    # 2. size of the output, normally
    #    2 (only the center, LabelType.CenterOnly),
    #    5 (center and shape of one ellipse, LabelType.InnerOnly),
    #    10 (center and shape of two ellipses, LabelType.InnerOuter)
    # 3. the direction for saving trained model, and the interval of saving
    # 4. the training and testing set
    print("Specify the model type:\n1: center only\n2: inner ellipse\n3: outer ellipse")
    mode = int(input("Mode = "))
    output_size = LabelType.CenterOnly
    if mode == 1:
        output_size = LabelType.CenterOnly
    elif mode == 2:
        output_size = LabelType.InnerOnly
    elif mode == 3:
        output_size = LabelType.InnerOuter

    Net = CoilNet(img_size=512, output_size=output_size)
    Net.learn_rate = 0.00001
    Net.l2_lambda = 0.0002
    Net.dropout_rate = 0.4
    Net.save_dir = "D:/01 自动上卷/Python"
    Net.save_prefix = "带卷定位"
    Net.save_interval = 5000
    Net.init_model()

    # specify the train and test set
    train_set = ImageSet("D:/TrainOuter", output_size=Net.output_size, img_size=Net.img_size)
    eval_set = train_set

    mode = input("Train(T) the model, or run(R) the model?")
    if mode == 't' or mode == 'T':
        Net.train(train_set, eval_set, epoch_cnt=50000, mini_batch=32)
    else:
        Net.load_model("D:/01 自动上卷/Python/2020-10-23-23-18.pt")   # load existing model to continue training
        Net.test_net(eval_set)

if __name__ == '__main__':
    main()
