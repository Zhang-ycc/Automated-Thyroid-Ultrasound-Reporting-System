import os
import pickle

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# image结节图像
def existence(image, mask):
    # image = cv.imread('example/Thyroid nodules/IM_0011.png', 0)  # 读取灰度图像
    print(image.shape)
    Image = cv.bitwise_and(image, image, mask=mask)
    print(Image.shape)
    # IMAGE = Image
    # Image = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)
    print(Image.shape)
    image = Image

    # 提取结晶特征
    # crystal_threshold = 140  # 结晶阈值
    X_test = []
    # gray_image = cv.cvtColor(IMAGE, cv.COLOR_BGR2GRAY)
    gray_image = Image
    features = gray_image.flatten()
    X_test.append(features)
    shapes = set(arr.shape for arr in X_test)
    if len(shapes) >= 1:
        X_test = [np.resize(arr, (786432,)) for arr in X_test]
    X_test = np.array(X_test)

    with open('./calcification/model.pkl', 'rb') as file:
        model = pickle.load(file)
    y_test_pred = model.predict(X_test)

    crystal_threshold = y_test_pred[0]
    # print(crystal_threshold)

    _, crystal_binary = cv.threshold(image, crystal_threshold, 255, cv.THRESH_BINARY)

    # 计算结晶和钙化区域的像素个数
    crystal_pixel_count = cv.countNonZero(crystal_binary)

    # 根据像素个数进行判断
    if crystal_pixel_count > 0:
        existence = 1
    else:
        existence = 0

    # 寻找结晶和钙化的轮廓
    crystal_contours, _ = cv.findContours(crystal_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 创建一个彩色图像用于绘制轮廓
    result = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # 绘制结晶的轮廓
    cv.drawContours(result, crystal_contours, -1, (0, 255, 0), 2)

    # 显示结果图像
    # cv.imwrite('example/existence/IM_0011.png', result)
    # cv.imshow('Result', result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return existence, result
