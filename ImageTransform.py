import cv2


"""
因为数据集图像不涉及nan值和inf值，因此直接使用了 cv2.warpPerspective获得旋转后的图像。
"""
import math
import cv2
import numpy as np

# 按带卷中心旋转，旋转角度选择不要太大，顺时针旋转角度为负，逆时针旋转角度为正，path是图片路径
rotate_angle = -25
path = "20211110224948-1478-0997-0217-0138-0055-1566-0921-0668-0431-0054.jpg"

def get_dstShape(image, transfor_Mat):
    """
    该函数获取旋转变换后图像的shape
    """
    height = image.shape[0]
    width = image.shape[1]

    x_coordinate = np.repeat([np.arange(width)], repeats=height, axis=0)
    y_coordinate = (np.repeat([np.arange(height)], repeats=width, axis=0)).T
    w_coordinate = np.ones((height, width), dtype=np.int16)

    x_coordinate = x_coordinate[:, :, np.newaxis]
    y_coordinate = y_coordinate[:, :, np.newaxis]
    w_coordinate = w_coordinate[:, :, np.newaxis]
    xyw_coordinate = np.concatenate((x_coordinate, y_coordinate, w_coordinate), axis=2)

    # 获得原图像的每个坐标透视变换后的坐标值
    xyw = xyw_coordinate.reshape((-1, 3))
    transformed_xyw = np.dot(transfor_Mat, xyw.T)
    transformed_X = transformed_xyw[0, :] / transformed_xyw[2, :]
    transformed_Y = transformed_xyw[1, :] / transformed_xyw[2, :]
    transformed_X = np.around(transformed_X).astype(np.int16)
    transformed_Y = np.around(transformed_Y).astype(np.int16)

    # 获得目标图像的shape
    dst_height = int(np.max(transformed_Y))
    dst_width = int(np.max(transformed_X))
    dst_shape = np.array([dst_width, dst_height]).astype(np.int16)

    return dst_shape


def get_tapeRoll_data(path):
    """
    该函数获取透视变换前带卷的中心、长轴、短轴和旋转角度
    """
    strlist = path.split('.')[0]
    strlist = strlist.split('-')
    tapeRoll_data = []
    for value in strlist:
        tapeRoll_data.append(value)

    return tapeRoll_data[0], int(tapeRoll_data[1]), int(tapeRoll_data[2]), int(tapeRoll_data[3]), int(tapeRoll_data[4]), int(tapeRoll_data[5])

def get_transfor_Mat(CenterX, CenterY):
    """
    须注意的是透视变换矩阵的a31和a32一般设置e-4数量级，这里按照经验自行选择了4个透视变换矩阵，
    """
    transfor_Mat = cv2.getRotationMatrix2D((int(CenterX), int(CenterY)), rotate_angle, 1.0)
    transfor_Mat =  np.array(transfor_Mat)
    transfor_Mat = np.float64([[transfor_Mat[0, :], transfor_Mat[1, :], [0.00005,   0.00005,   1]],
                               [transfor_Mat[0, :], transfor_Mat[1, :], [0.00005,   -0.00005,  1]],
                               [transfor_Mat[0, :], transfor_Mat[1, :], [-0.00005, -0.00005, 1]],
                               [transfor_Mat[0, :], transfor_Mat[1, :], [-0.00005,  0.00005,   1]]])

    return transfor_Mat

def get_transorforAxis(transfor_Mat, CenterX, CenterY, majorAxis, minorAxis, angle):
    """
    angle尽量保证在180°以内，一般来说带卷的旋转角度就在180°以内，此处主要利用了三角形的正余弦角度求解原长轴短轴的边界点，
    再利用透视变换矩阵映射到新的图像上，进而求得变换后图像的长轴和短轴
    """
    cosValue = math.cos(np.pi * (angle / 180))
    sinValue = math.sin(np.pi * (angle / 180))
    majorAxis_point1 = np.array([[CenterX + majorAxis * cosValue / 2], [CenterY + majorAxis * sinValue / 2], [1]])
    majorAxis_point2 = np.array([[CenterX - majorAxis * cosValue / 2], [CenterY - majorAxis * sinValue / 2], [1]])
    minorAxis_point1 = np.array([[CenterX + minorAxis * sinValue / 2], [CenterY - minorAxis * cosValue / 2], [1]])
    minorAxis_point2 = np.array([[CenterX - minorAxis * sinValue / 2], [CenterY + minorAxis * cosValue / 2], [1]])

    transforMajorAxis_point1 = np.dot(transfor_Mat, majorAxis_point1)
    transforMajorAxis_point2 = np.dot(transfor_Mat, majorAxis_point2)
    transforMinorAxis_point1 = np.dot(transfor_Mat, minorAxis_point1)
    transforMinorAxis_point2 = np.dot(transfor_Mat,  minorAxis_point2)

    transforMajorAxis_point1 = np.float64([transforMajorAxis_point1[0] / transforMajorAxis_point1[2], transforMajorAxis_point1[1] / transforMajorAxis_point1[2]])
    transforMajorAxis_point2 = np.float64([transforMajorAxis_point2[0] / transforMajorAxis_point2[2], transforMajorAxis_point2[1] / transforMajorAxis_point2[2]])
    transforMinorAxis_point1 = np.float64([transforMinorAxis_point1[0] / transforMinorAxis_point1[2], transforMinorAxis_point1[1] / transforMinorAxis_point1[2]])
    transforMinorAxis_point2 = np.float64([transforMinorAxis_point2[0] / transforMinorAxis_point2[2], transforMinorAxis_point2[1] / transforMinorAxis_point2[2]])

    point_subtractMajorAxis = transforMajorAxis_point1 - transforMajorAxis_point2
    transforMajorAxis = math.sqrt(pow(point_subtractMajorAxis[0], 2) + pow(point_subtractMajorAxis[1], 2))
    point_subtractMinorAxis = transforMinorAxis_point1 - transforMinorAxis_point2
    transforMinorAxis = math.sqrt(pow(point_subtractMinorAxis[0], 2) + pow(point_subtractMinorAxis[1], 2))

    return int(transforMajorAxis+0.5), int(transforMinorAxis+0.5)

def get_transorforCenter(transfor_Mat, CenterX, CenterY):

    point = np.array([CenterX, CenterY, 1])
    transfor_point = np.dot(transfor_Mat, point.T)
    transfor_CenterX = int(transfor_point[0] / transfor_point[2] + 0.5)
    transfor_CenterY = int(transfor_point[1] / transfor_point[2] + 0.5)

    return int(transfor_CenterX+0.5),int(transfor_CenterY+0.5)

def image_enhance(path):
    """
    该函数用于实现图像增强
    """
    gray_image = cv2.imread(path, 0)
    date, CenterX, CenterY, majorAxis, minorAxis, angle = get_tapeRoll_data(path)

    transfor_Mat = get_transfor_Mat(CenterX, CenterY)

    # 求仿射变换后图像的角度
    Angle_ = int(-rotate_angle + int(angle))
    transformed_angle = str((Angle_ - 180) if (Angle_ > 180 or Angle_ == 180) else Angle_).zfill(3)

    for i in range(len(transfor_Mat)):
        # 求仿射变换后的图像
        dst_shape = get_dstShape(gray_image, transfor_Mat[i, :])
        transformedImage = cv2.warpPerspective(gray_image, transfor_Mat[i, :], dst_shape, borderValue=135)

        transfor_CenterX, transfor_CenterY =  get_transorforCenter(transfor_Mat[i, :], CenterX, CenterY)
        transfor_majorAxis, transfor_minorAxis = get_transorforAxis(transfor_Mat[i, :], CenterX, CenterY, majorAxis, minorAxis, angle)

        #长轴大于230，以透视变换后的中心点为基准来截取transformedImage的大小为512*512的图像，可能显示不完全，故直接舍去，
        if transfor_majorAxis > 230:
            continue

        # 中心随机化，提高泛化能力
        randNumX = np.random.randint(-10,10)
        randNumY = np.random.randint(-10, 10)
        CenterX_left = transfor_CenterX + randNumX - 256
        CenterY_top = transfor_CenterY + randNumX - 256
        CenterX_right = transfor_CenterX + randNumY + 256
        CenterY_bottom = transfor_CenterY + randNumY + 256

        transfor_majorAxis = str(transfor_majorAxis).zfill(4)
        transfor_minorAxis = str(transfor_minorAxis).zfill(4)
        transforCentorX = str(256 - randNumX).zfill(4)
        transforCentorY = str(256 - randNumY).zfill(4)
        new_path = date + "-" + transforCentorX + "-" + transforCentorY + "-" + transfor_majorAxis + "-" + transfor_minorAxis + "-" + transformed_angle + ".jpg"
        #cv2.imwrite(new_path, transformedImage[CenterY_top:CenterY_bottom, CenterX_left:CenterX_right])
        cv2.imwrite(new_path, transformedImage)
        cv2.namedWindow("img", 0)
        cv2.imshow("img", transformedImage)
        cv2.waitKey(100)

def main():
    image_enhance(path)

if __name__ == "__main__":
    m = cv2.getRotationMatrix2D((0, 0), -25, 0.5)
    print(m)
    m = cv2.getRotationMatrix2D((1, 1), -25, 0.5)
    print(m)

