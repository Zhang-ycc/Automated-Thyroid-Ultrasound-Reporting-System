import os
import pickle

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # image = cv.imread('example/Thyroid nodules/IM_0011.png', 0)  # 读取灰度图像
    # image = cv.GaussianBlur(image, (5, 5), 0)  # 高斯平滑滤波
    mask = cv.imread('example/Dataset002_SJTUThyroid/labelsTr02/SJTU_026.png', 0)  # 读取灰度图像

    image = cv.imread('example/Dataset002_SJTUThyroid/imagesTr/SJTU_026_0000.png')  # 读取灰度图像

    height, width = image.shape[:2]
    # 将掩膜应用于原始图像，提取椭圆内的图像
    Image = cv.bitwise_and(image, image, mask=mask)
    IMAGE = Image
    Image = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)
    image = Image

    # 提取结晶特征
    # crystal_threshold = 140  # 结晶阈值
    X_test = []
    gray_image = cv.cvtColor(IMAGE, cv.COLOR_BGR2GRAY)
    features = gray_image.flatten()
    X_test.append(features)
    shapes = set(arr.shape for arr in X_test)
    if len(shapes) >= 1:
        X_test = [np.resize(arr, (786432,)) for arr in X_test]
    X_test = np.array(X_test)

    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    y_test_pred = model.predict(X_test)

    crystal_threshold = y_test_pred[0]
    print(crystal_threshold)

    _, crystal_binary = cv.threshold(image, crystal_threshold, 255, cv.THRESH_BINARY)

    # # 提取钙化特征
    # calcification_threshold = 200  # 钙化阈值
    # _, calcification_binary = cv.threshold(image, calcification_threshold, 255, cv.THRESH_BINARY)

    # 计算结晶和钙化区域的像素个数
    crystal_pixel_count = cv.countNonZero(crystal_binary)
    print(crystal_pixel_count)
    # calcification_pixel_count = cv.countNonZero(calcification_binary)
    # print(calcification_pixel_count)

    # 根据像素个数进行判断
    if crystal_pixel_count > 0:
        print("结晶存在")
    else:
        print("结晶不存在")

    # if calcification_pixel_count > 0:
    #     print("钙化存在")
    # else:
    #     print("钙化不存在")

    # 寻找结晶和钙化的轮廓
    crystal_contours, _ = cv.findContours(crystal_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # calcification_contours, _ = cv.findContours(calcification_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 创建一个彩色图像用于绘制轮廓
    result = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    # 绘制结晶的轮廓
    cv.drawContours(result, crystal_contours, -1, (0, 255, 0), 2)

    # 绘制钙化的轮廓
    # cv.drawContours(result, calcification_contours, -1, (0, 0, 255), 2)

    # 显示结果图像
    # cv.imwrite('example/existence/3.png', result)
    cv.imshow('Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # image = cv.imread('example/Thyroid nodules/26(1).png', 0)
    # # 实际：调用结节轮廓，获得轮廓内图像
    # threshold = 100
    # threshold2 = 1500
    #
    # # 应用高斯滤波平滑图像
    # smoothed = cv.GaussianBlur(image, (5, 5), 0)
    #
    # # 应用自适应阈值化
    # _, thresholded = cv.threshold(smoothed, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #
    # # 进行形态学操作（开运算）
    # kernel = np.ones((3, 3), np.uint8)
    # opened = cv.morphologyEx(thresholded, cv.MORPH_OPEN, kernel)
    #
    # # 寻找轮廓
    # contours, _ = cv.findContours(opened, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #
    # # 调用甲状腺结节边缘信息函数
    #
    # # 绘制轮廓和结晶标记
    # result = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # cv.drawContours(result, contours, -1, (0, 0, 255), 2)
    #
    # for contour in contours:
    #     # 计算轮廓的面积
    #     area = cv.contourArea(contour)
    #     print(area)
    #     if threshold < area < threshold2:
    #         # 绘制结晶标记（绿色矩形框）
    #         print("crystallization exist")
    #         x, y, w, h = cv.boundingRect(contour)
    #         cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    # # 显示结果图像
    # cv.imwrite('example/crystallization/26.png', result)
    # cv.imshow('Result', result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
