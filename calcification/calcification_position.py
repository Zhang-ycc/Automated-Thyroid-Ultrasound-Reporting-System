import os
import pickle

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# image甲状腺图像
def CalcificationPosition(mask, image):
    mask = mask * 255

    height, width = image.shape[:2]
    Image = cv.bitwise_and(image, image, mask=mask)
    # Image = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)

    Image = cv.GaussianBlur(Image, (5, 5), 0)

    X_test = []
    # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    features = Image.flatten()
    X_test.append(features)
    shapes = set(arr.shape for arr in X_test)
    if len(shapes) >= 1:
        X_test = [np.resize(arr, (786432,)) for arr in X_test]
    X_test = np.array(X_test)

    with open('./calcification/model.pkl', 'rb') as file:
        model = pickle.load(file)
    y_test_pred = model.predict(X_test)

    calcification_threshold = y_test_pred[0]
    print(calcification_threshold)

    _, calcification_binary = cv.threshold(Image, calcification_threshold, 255, cv.THRESH_BINARY)

    # 计算钙化区域的像素个数
    calcification_pixel_count = cv.countNonZero(calcification_binary)
    # print(calcification_pixel_count)

    # 寻找钙化的轮廓
    calcification_contours, _ = cv.findContours(calcification_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 创建一个彩色图像用于绘制轮廓
    result = cv.cvtColor(Image, cv.COLOR_GRAY2BGR)

    # 绘制钙化的轮廓
    cv.drawContours(result, calcification_contours, -1, (0, 0, 255), 2)
    # print(len(calcification_contours))

    # 输出钙化的位置坐标
    for calcification_contour in calcification_contours:
        # 获取边界框坐标
        x, y, w, h = cv.boundingRect(calcification_contour)
        print("钙化位置坐标：({},{})".format(x, y), " 宽：", w, " 高：", h)

    # # 显示结果图像
    # cv.imwrite('example/position/IM_0011.png', result)
    # cv.imshow('Result', result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return result
