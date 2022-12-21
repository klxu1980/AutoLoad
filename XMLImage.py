import cv2
import os
import glob
import random
import numpy as np
import datetime
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from CoilImageProc import *


class XmlImage(object):
    def __init__(self, file_name, grey_scaled=False, pre_proc=False):
        self.file_name = file_name
        self.image = None
        self.image_file = None
        self.grey_scaled = grey_scaled
        self.pre_proc = pre_proc
        self.names = []           # the name of the objects
        self.bboxes = []          # anchor box of the objects
        self.centers = []         # centers of the objects
        self.ellipses = []        # ellipses (long and short axises), its center stored in centers, angle in directions
        self.directions = []

        xml = ET.parse(file_name)
        root = xml.getroot()
        for node in root:
            if node.tag == "filename":
                image_file = node.text
                img_dir = os.path.dirname(file_name)
                self.image_file = img_dir + "\\" + image_file if len(img_dir) else image_file
                if os.path.exists(self.image_file):
                    cv_img = cv_read(self.image_file)
                    image = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY) if self.grey_scaled else cv_img

                    if image is not None and pre_proc:
                        image = edge_image(image)
                    self.image = image
            elif node.tag == "object":
                self.__load_labels(node)
        if self.image is None:
            print(self.image_file + " cannot be loaded")

    def __load_labels(self, node):
        name = None
        bbox = (-1, -1, -1, -1)
        ellipse = (-1, -1)
        center = (-1, -1)
        direction = -1
        for child in node:
            if child.tag == "name":
                name = child.text
            elif child.tag == "bndbox":
                left = top = right = bottom = 0
                for pos in child:
                    if pos.tag == "xmin":
                        left = int(pos.text)
                    elif pos.tag == "xmax":
                        right = int(pos.text)
                    elif pos.tag == "ymin":
                        top = int(pos.text)
                    elif pos.tag == "ymax":
                        bottom = int(pos.text)
                bbox = (left, top, right, bottom)
            elif child.tag == "ellipse":
                long_axis = short_axis = 0
                for pos in child:
                    if pos.tag == "long_axis":
                        long_axis = int(pos.text)
                    elif pos.tag == "short_axis":
                        short_axis = int(pos.text)
                ellipse = (long_axis, short_axis)
            elif child.tag == "center":
                x = y = 0
                for pos in child:
                    if pos.tag == "x":
                        x = int(pos.text)
                    elif pos.tag == "y":
                        y = int(pos.text)
                center = (x, y)
            elif child.tag == "direction":
                direction = int(child.text)

        if name is not None:
            self.names.append(name)
            self.bboxes.append(bbox)
            self.ellipses.append(ellipse)
            self.centers.append(center)
            self.directions.append(direction)

    def __write_labels(self, root):
        for i in range(len(self.names)):
            obj_node = Element("object")
            root.append(obj_node)

            name = Element("name")
            obj_node.append(name)
            name.text = self.names[i]

            bbox = Element("bndbox")
            obj_node.append(bbox)
            xmin = Element("xmin")
            bbox.append(xmin)
            xmin.text = "%d" % self.bboxes[i][0]
            ymin = Element("ymin")
            bbox.append(ymin)
            ymin.text = "%d" % self.bboxes[i][1]
            xmax = Element("xmax")
            bbox.append(xmax)
            xmax.text = "%d" % self.bboxes[i][2]
            ymax = Element("ymax")
            bbox.append(ymax)
            ymax.text = "%d" % self.bboxes[i][3]

            ellipse = Element("ellipse")
            obj_node.append(ellipse)
            long_axis = Element("long_axis")
            ellipse.append(long_axis)
            long_axis.text = "%d" % self.ellipses[i][0]
            short_axis = Element("short_axis")
            ellipse.append(short_axis)
            short_axis.text = "%d" % self.ellipses[i][1]

            center = Element("center")
            obj_node.append(center)
            x = Element("x")
            center.append(x)
            x.text = "%d" % self.centers[i][0]
            y = Element("y")
            center.append(y)
            y.text = "%d" % self.centers[i][1]

            direction = Element("direction")
            obj_node.append(direction)
            direction.text = "%d" % self.directions[i]

    def show_labels(self):
        img = self.image.copy()
        for _, bbox in enumerate(self.bboxes):
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] >= 0 and bbox[3] >= 0:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(20, 255, 20))

        for _, center in enumerate(self.centers):
            if center[0] >= 0 and center[1] >= 0:
                cv2.line(img, (center[0] - 10, center[1]), (center[0] + 10, center[1]), color=(20, 255, 20))
                cv2.line(img, (center[0], center[1] - 10), (center[0], center[1] + 10), color=(20, 255, 20))

        return img

    def save(self):
        cv_write(self.image_file, self.image)

        xml = ET.parse(self.file_name)
        root = xml.getroot()
        for node in root:
            if node.tag == "object":
                root.remove(node)
        self.__write_labels(root)
        ET.ElementTree(root).write(self.file_name, encoding='utf-8', xml_declaration=True)

    def gen_subimages_around_centers(self, size, random_range, count):
        sub_images = []
        centers = []

        random.seed()
        for _, center in enumerate(self.centers):
            for i in range(count):
                # decide the center of the sub-image randomly around the center of the inner ellipse
                x_offset = random.randint(-random_range, random_range)
                y_offset = random.randint(-random_range, random_range)
                x_center = center[0] + x_offset
                y_center = center[1] + y_offset

                # the position of the sub-image
                src_left = int(max(0, x_center - size[0] / 2))
                src_top = int(max(0, y_center - size[1] / 2))
                src_right = int(min(self.image.shape[1], x_center + size[0] / 2))
                src_bottom = int(min(self.image.shape[0], y_center + size[1] / 2))
                src_width = src_right - src_left
                src_height = src_bottom - src_top

                # discard the uncompact sub-image
                if src_width < size[0] or src_height < size[1]:
                    continue

                sub_img = self.image[src_top: src_bottom, src_left: src_right]
                sub_images.append(sub_img)

                center_x = int(size[0] / 2 - x_offset)
                center_y = int(size[1] / 2 - y_offset)
                centers.append((center_x, center_y))

        return sub_images, centers

    def resize(self, image_size):
        resize_kx = image_size[0] / float(self.image.shape[1])
        resize_ky = image_size[1] / float(self.image.shape[0])

        for i in range(len(self.bboxes)):
            self.bboxes[i] = (int(self.bboxes[i][0] * resize_kx),
                              int(self.bboxes[i][1] * resize_ky),
                              int(self.bboxes[i][2] * resize_kx),
                              int(self.bboxes[i][3] * resize_ky))

        for i in range(len(self.centers)):
            self.centers[i] = (int(self.centers[i][0] * resize_kx),
                               int(self.centers[i][1] * resize_ky))

        self.image = cv2.resize(self.image, image_size)


if __name__ == '__main__':
    idx = 1
    for file_name in glob.glob("E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷初始定位\\*.xml"):
        labeled_image = XmlImage(file_name, grey_scaled=False, pre_proc=False)
        if labeled_image.image is None:
            continue
        labeled_image.resize((512, 512))

        save_name = "E:\\01 我的设计\\05 智通项目\\04 自动上卷\\带卷定位训练\\InitPos\\" + \
                    datetime.datetime.now().strftime("%Y%m%d%H%M") + \
                    ("%d-%d-%d.jpg" % (idx, labeled_image.centers[0][0], labeled_image.centers[0][1]))
        cv_write(save_name, labeled_image.image)

        continue

        img = labeled_image.show_labels()
        cv2.putText(img, file_name, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 255, 20), 1)
        cv2.imshow("image", img)
        if cv2.waitKey(0) == ord('e'):
            break

        """
        sub_imgs, centers = labeled_image.gen_subimages_around_centers((256, 256), 32, 2)
        for i, img in enumerate(sub_imgs):
            x = centers[i][0]
            y = centers[i][1]

            #cv2.line(img, (x - 3, y), (x + 3, y), color=(20, 255, 20))
            #cv2.line(img, (x, y - 3), (x, y + 3), color=(20, 255, 20))
            cv2.rectangle(img, (x - 8, y - 8), (x + 8, y + 8), color=(20, 255, 20))

            cv2.imshow("sub-image", img)
            cv2.waitKey(0)
        """

