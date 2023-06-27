import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mask = cv.imread('example/Dataset002_SJTUThyroid/labelsTr01/SJTU_001.png', 0)  # 读取灰度图像
    mask = mask * 255

    image = cv.imread('example/Dataset002_SJTUThyroid/imagesTr/SJTU_001_0000.png')  # 读取灰度图像
    height, width = image.shape[:2]
    # 将掩膜应用于原始图像，提取椭圆内的图像
    Image = cv.bitwise_and(image, image, mask=mask)
    Image = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)

    Image = cv.GaussianBlur(Image, (5, 5), 0)  # 高斯平滑滤波

    # 提取钙化特征
    calcification_threshold = 120  # 钙化阈值
    _, calcification_binary = cv.threshold(Image, calcification_threshold, 255, cv.THRESH_BINARY)

    # 计算钙化区域的像素个数
    calcification_pixel_count = cv.countNonZero(calcification_binary)
    print(calcification_pixel_count)

    # 寻找钙化的轮廓
    calcification_contours, _ = cv.findContours(calcification_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 创建一个彩色图像用于绘制轮廓
    # print(Image.shape)
    # print(image.shape)
    # result = cv.cvtColor(Image, cv.COLOR_GRAY2BGR)
    result = image

    # 绘制钙化的轮廓
    # cv.drawContours(result, calcification_contours, -1, (0, 0, 255), 2)

    edge = cv.imread('example/Dataset002_SJTUThyroid/labelsTr02/SJTU_001.png', 0)  # 读取灰度图像
    _, threshold = cv.threshold(edge, 127, 255, cv.THRESH_BINARY)
    nodules_contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 创建一个彩色图像用于绘制结节轮廓
    nodule_result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    nodule_result = cv.cvtColor(nodule_result, cv.COLOR_GRAY2BGR)

    # 绘制结节的轮廓
    cv.drawContours(result, nodules_contours, -1, (0, 0, 0), 2)

    # # 遍历所有的钙化轮廓
    # for calcification_contour in calcification_contours:
    #     # 判断钙化轮廓与结节轮廓是否相交
    #     is_intersection = False
    #     for nodule_contour in nodules_contours:
    #         calcification_mask = np.zeros_like(mask)  # 创建与mask相同大小的全0图像作为钙化轮廓的掩膜
    #         nodule_mask = np.zeros_like(mask)  # 创建与mask相同大小的全0图像作为结节轮廓的掩膜
    #         cv.drawContours(calcification_mask, [calcification_contour], -1, 255, thickness=cv.FILLED)  # 绘制钙化轮廓
    #         cv.drawContours(nodule_mask, [nodule_contour], -1, 255, thickness=cv.FILLED)  # 绘制结节轮廓
    #
    #         intersection = cv.bitwise_and(calcification_mask, nodule_mask)
    #         if cv.countNonZero(intersection) > 0:
    #             is_intersection = True
    #             break
    #
    #     # 根据相交与否进行标记
    #     if is_intersection:
    #         cv.drawContours(result, [calcification_contour], -1, (0, 0, 255), 2)
    #     else:
    #         cv.drawContours(result, [calcification_contour], -1, (255, 0, 0), 2)

    i = 0
    # 遍历所有的钙化轮廓
    for calcification_contour in calcification_contours:
        # 判断钙化轮廓与结节轮廓是否相交
        is_intersection = False
        is_inside = False  # 用于判断钙化轮廓是否在结节内部
        for nodule_contour in nodules_contours:
            calcification_mask = np.zeros_like(mask)  # 创建与mask相同大小的全0图像作为钙化轮廓的掩膜
            nodule_mask = np.zeros_like(mask)  # 创建与mask相同大小的全0图像作为结节轮廓的掩膜
            cv.drawContours(calcification_mask, [calcification_contour], -1, 255, thickness=cv.FILLED)  # 绘制钙化轮廓
            cv.drawContours(nodule_mask, [nodule_contour], -1, 255, thickness=cv.FILLED)  # 绘制结节轮廓

            intersection = cv.bitwise_and(calcification_mask, nodule_mask)
            if cv.countNonZero(intersection) > 0:
                is_intersection = True
                break

        if cv.contourArea(calcification_contour) > 0 and len(nodules_contours) >= 1:
            is_inside = True
            for point in calcification_contour:
                x = int(point[0][0])
                y = int(point[0][1])
                # print(x, y)
                position = cv.pointPolygonTest(nodules_contours[0], (x, y), measureDist=False)
                if position < 0:
                    is_inside = False  # 有任意一个点不在nodules_contours内部，则假设不成立
                    break

        # if is_intersection:
        #     print(1, "相交")
        #     # cv.drawContours(result, [calcification_contour], -1, (0, 0, 255), 2)
        # elif is_inside:
        #     print(2, "内部")
        #     # cv.drawContours(result, [calcification_contour], -1, (0, 255, 0), 2)
        # else:
        #     print(3, "外部")
        #     # cv.drawContours(result, [calcification_contour], -1, (255, 0, 0), 2)
        if is_inside:
            # print(1, "内部")
            cv.drawContours(result, [calcification_contour], -1, (0, 0, 255), 2)
        elif is_intersection:
            # print(2, "相交")
            cv.drawContours(result, [calcification_contour], -1, (0, 255, 0), 2)
        else:
            # print(3, "外部")
            cv.drawContours(result, [calcification_contour], -1, (255, 0, 0), 2)

        # if i == 18:
        #     cv.drawContours(result, [calcification_contour], -1, (0, 0, 255), 2)
        #     print(is_inside)
        i += 1

    # 显示结果图像
    cv.imwrite('example/compare/2.png', result)
    cv.imshow('Result', result)
    # cv.imshow('Calcification Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
