import os
import math
import numpy as np
from scipy import ndimage
import cv2


class HorizontalCorrection:
    def __init__(self):
        self.rotate_vector = np.array([0, 1])  # 图片中地面的法线向量
        self.rotate_theta = 0  # 旋转的角度

    def process(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 二值化
        edges = cv2.Canny(gray, 350, 400, apertureSize=3)  # canny算子

        # 霍夫变换
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
        sum = 0
        count = 0
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                if x2 != x1:
                    t = float(y2 - y1) / (x2 - x1)
                    if t <= np.pi / 5 and t >= - np.pi / 5:
                        rotate_angle = math.degrees(math.atan(t))
                        sum += rotate_angle
                        count += 1
                        # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if count == 0:
            avg_rotate_angle = 0
        else:
            avg_rotate_angle = sum / count
        rotate_img = ndimage.rotate(img, avg_rotate_angle)  # 逆时针旋转

        rotate_vector = self.count_rotate_vector(avg_rotate_angle)
        return rotate_img, avg_rotate_angle, rotate_vector

    def count_rotate_vector(self, rotate_theta):
        rotate_vector = np.array([0, 1])
        v1_new = (rotate_vector[0] * np.cos(rotate_theta / 180)) - \
                 (rotate_vector[1] * np.sin(rotate_theta / 180))
        v2_new = (rotate_vector[1] * np.cos(rotate_theta / 180)) + \
                 (rotate_vector[0] * np.sin(rotate_theta / 180))
        rotate_vector = np.array([v1_new, v2_new])


if __name__ == '__main__':
    horizontal_correction = HorizontalCorrection()
    data_root = "/Users/xiangyuwang/Desktop/户口图片"
    src = os.path.join(data_root, "w202210261148271563.jpg")
    rotate_img, rotate_theta, rotate_vector = horizontal_correction.process(src)
    print(rotate_theta)
    cv2.imwrite('1.png', rotate_img)
