import os
import pickle

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont, Image

ShapeTypes  = set()

def compute_curvature(points, step_size):
    P_prev = np.roll(points, 1, axis=0)
    P_next = np.roll(points, -1, axis=0)
    curvature = (P_prev - 2 * points + P_next) / (step_size ** 2)
    return curvature

def add_shape(shape):
    if shape not in ShapeTypes:
        ShapeTypes.add(shape)

if __name__ == '__main__':
    type = 0
    mask = cv.imread('example/Dataset002_SJTUThyroid/labelsTr01/SJTU_028.png', 0)  # 读取灰度图像
    mask = mask * 255

    image = cv.imread('example/Dataset002_SJTUThyroid/imagesTr/SJTU_028_0000.png')  # 读取灰度图像
    height, width = image.shape[:2]
    # 将掩膜应用于原始图像，提取椭圆内的图像
    IMAGE = cv.bitwise_and(image, image, mask=mask)
    IMAGE = cv.cvtColor(IMAGE, cv.COLOR_BGR2GRAY)

    IMAGE = cv.GaussianBlur(IMAGE, (5, 5), 0)  # 高斯平滑滤波

    # 提取钙化特征
    # calcification_threshold = 120  # 钙化阈值
    X_test = []
    # gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    features = IMAGE.flatten()
    X_test.append(features)
    shapes = set(arr.shape for arr in X_test)
    if len(shapes) >= 1:
        X_test = [np.resize(arr, (786432,)) for arr in X_test]
    X_test = np.array(X_test)

    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    y_test_pred = model.predict(X_test)

    calcification_threshold = y_test_pred[0]
    # print(calcification_threshold)

    _, calcification_binary = cv.threshold(IMAGE, calcification_threshold, 255, cv.THRESH_BINARY)

    # 计算钙化区域的像素个数
    calcification_pixel_count = cv.countNonZero(calcification_binary)
    # print(calcification_pixel_count)

    # 寻找钙化的轮廓
    calcification_contours, _ = cv.findContours(calcification_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    result = image

    edge = cv.imread('example/Dataset002_SJTUThyroid/labelsTr02/SJTU_028.png', 0)  # 读取灰度图像
    _, threshold = cv.threshold(edge, 127, 255, cv.THRESH_BINARY)
    nodules_contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 创建一个彩色图像用于绘制结节轮廓
    nodule_result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    nodule_result = cv.cvtColor(nodule_result, cv.COLOR_GRAY2BGR)

    # 绘制结节的轮廓
    cv.drawContours(result, nodules_contours, -1, (0, 0, 0), 2)

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

        if is_inside:
            # print(1, "内部")
            type = 1
            # cv.drawContours(result, [calcification_contour], -1, (0, 0, 255), 2)
        elif is_intersection:
            # print(2, "相交")
            type = 2
            # cv.drawContours(result, [calcification_contour], -1, (0, 255, 0), 2)
        else:
            # print(3, "外部")
            type = 3
            # cv.drawContours(result, [calcification_contour], -1, (255, 0, 0), 2)

        # if i == 18:
        #     cv.drawContours(result, [calcification_contour], -1, (0, 0, 255), 2)
        #     print(is_inside)
        i += 1

        # print("type: ", type)
        if type == 1:
            # 斑点钙化
            shape = ""
            threshold = 3
            kernel = np.ones((3, 3), np.uint8)

            # 创建与calcification_contour相同大小的全0图像作为钙化轮廓的掩膜
            calcification_mask = np.zeros_like(calcification_binary)
            # 绘制当前calcification_contour为白色区域（255）
            cv.drawContours(calcification_mask, [calcification_contour], -1, 255, thickness=cv.FILLED)

            erosion = cv.erode(calcification_mask, kernel, iterations=1)
            overlap = cv.bitwise_and(calcification_mask, erosion)
            overlap_pixel_count = cv.countNonZero(overlap)
            print(overlap_pixel_count)
            if overlap_pixel_count > threshold:
                calcification_masked = cv.bitwise_and(calcification_mask, calcification_mask)  # 通过与钙化轮廓掩膜相与获取结节内的钙化部分
                calcification_contours_inner, _ = cv.findContours(calcification_masked, cv.RETR_EXTERNAL,
                                                                  cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(result, calcification_contours_inner, -1, (0, 0, 128), 2)
                shape = "内部钙化（斑点钙化）"
                # print(shape)

            # 明亮钙化
            threshold_mean = 200  # 平均像素值阈值
            threshold_std = 30  # 像素值标准差阈值
            step_size = 1

            # 提取钙化区域的像素值
            calcification_pixels = IMAGE[np.where(calcification_mask > 0)]

            # 计算钙化区域的平均像素值和像素值标准差
            mean_value = np.mean(calcification_pixels)
            std_value = np.std(calcification_pixels)

            # 判断明亮钙化
            if mean_value > threshold_mean and std_value > threshold_std:
                cv.drawContours(result, [calcification_contour], -1, (128, 128, 255), 2)
                shape = "内部钙化（明亮钙化）"
                # print(shape)

            if len(calcification_contour) >= 5:
                # 弧形钙化
                curvature_threshold = 3  # 弧度阈值
                length_width_ratio_threshold = 0.6  # 长宽比阈值

                # 计算钙化轮廓的凸包
                hull = cv.convexHull(calcification_contour)

                # 计算钙化轮廓的曲率
                calcification_points = calcification_contour.squeeze()
                curvature = compute_curvature(calcification_points, step_size)
                # print("curvature:", curvature)

                # 计算钙化轮廓的长度与宽度比
                _, (major_axis, minor_axis), _ = cv.fitEllipse(calcification_contour)
                length_width_ratio = major_axis / minor_axis
                # print("length_width_ratio:", length_width_ratio)

                # 判断弧形钙化
                if np.max(curvature) > curvature_threshold and length_width_ratio > length_width_ratio_threshold:
                    cv.drawContours(result, [calcification_contour], -1, (0, 0, 255), 2)
                    shape = "内部钙化（弧形钙化）"

            # print(shape)
            add_shape(shape)

        elif type == 2:
            step_size = 1
            curvature_threshold = 3
            threshold = 0.7
            threshold2 = 0.2
            # 计算钙化轮廓的凸包
            hull = cv.convexHull(calcification_contour)

            # 拟合椭圆到钙化轮廓
            ellipse = cv.fitEllipse(calcification_contour)

            # 计算凸包和椭圆的面积比
            hull_area = cv.contourArea(hull)
            ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1]
            area_ratio = hull_area / ellipse_area
            # print(area_ratio)

            # 计算钙化轮廓的曲率
            calcification_points = calcification_contour.squeeze()
            curvature = compute_curvature(calcification_points, step_size)
            # print(curvature)

            # 判断钙化形状
            if area_ratio > threshold and np.max(curvature) < curvature_threshold:
                cv.drawContours(result, [calcification_contour], -1, (0, 255, 0), 2)
                shape = "边缘钙化（环状）"
            elif area_ratio > threshold2:
                cv.drawContours(result, [calcification_contour], -1, (0, 128, 0), 2)
                shape = "边缘钙化（弧形）"
            else:
                cv.drawContours(result, [calcification_contour], -1, (50, 205, 50), 2)
                shape = "边缘钙化（点状）"

            # print(shape)
            add_shape(shape)

        elif type == 3:
            step_size = 1
            curvature_threshold = 3
            threshold = 0.7
            threshold2 = 0.2
            # 计算钙化轮廓的凸包
            if len(calcification_contour) < 5:
                cv.drawContours(result, [calcification_contour], -1, (128, 0, 0), 2)
                shape = "外部钙化（点状）"
            else:
                hull = cv.convexHull(calcification_contour)

                # 拟合椭圆到钙化轮廓
                ellipse = cv.fitEllipse(calcification_contour)

                # 计算凸包和椭圆的面积比
                hull_area = cv.contourArea(hull)
                ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1]
                area_ratio = hull_area / ellipse_area
                # print(area_ratio)

                # 计算钙化轮廓的曲率
                calcification_points = calcification_contour.squeeze()
                curvature = compute_curvature(calcification_points, step_size)
                # print(curvature)

                # 判断钙化形状
                if area_ratio > threshold and np.max(curvature) < curvature_threshold:
                    cv.drawContours(result, [calcification_contour], -1, (64, 0, 0), 2)
                    shape = "外部钙化（环状）"
                elif area_ratio > threshold2:
                    cv.drawContours(result, [calcification_contour], -1, (255, 0, 0), 2)
                    shape = "外部钙化（弧形）"
                else:
                    cv.drawContours(result, [calcification_contour], -1, (128, 0, 0), 2)
                    shape = "外部钙化（点状）"

            # print(shape)
            add_shape(shape)

    colors = [(0, 0, 128), (128, 128, 255), (0, 0, 255), (0, 255, 0), (0, 128, 0), (50, 205, 50), (128, 0, 0),
              (64, 0, 0), (255, 0, 0)]  # 标签的颜色列表
    labels = ['内部钙化（斑点钙化）', '内部钙化（明亮钙化）', '内部钙化（弧形钙化）', '边缘钙化（环状）', '边缘钙化（弧形）',
              '边缘钙化（点状）', '外部钙化（点状）', '外部钙化（环状）', '外部钙化（弧形）']  # 标签的文字说明列表

    label_size = min(result.shape[0], result.shape[1]) // 40  # 标签区域的大小
    margin = label_size // 2  # 标签区域与边界的间距
    # font_path = 'SimSun.ttf'
    # font = ImageFont.truetype(font='SimSun.ttf', size=36)


    for i in range(9):
        start_x = margin  # 当前标签的起始x坐标
        end_x = start_x + label_size  # 当前标签的结束x坐标
        start_y = result.shape[0] - 5 * margin - (i + 1) * label_size  # 当前标签的起始y坐标
        end_y = start_y + label_size  # 当前标签的结束y坐标
        result[start_y:end_y, start_x:end_x] = colors[i]  # 在原图上绘制当前标签的颜色区域
        # cv.putText(result, labels[i], (end_x + margin, end_y - margin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
        #            1, cv.LINE_AA)  # 绘制当前标签的文字说明
        # cv.putText(result, labels[i], (end_x + margin, end_y - margin), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
        #            1, cv.LINE_AA, False)  # 绘制当前标签的文字说明
        pilimg = Image.fromarray(result)
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        font = ImageFont.truetype("SimSun.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        draw.text((end_x + margin, end_y - 2 * margin), labels[i], (255, 255, 255), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        result = cv.cvtColor(np.array(pilimg), cv.COLOR_RGB2BGR)

    print(ShapeTypes)

    # 显示结果图像
    # cv.imwrite('example/compare/2.png', result)
    cv.imshow('Result', result)
    # cv.imshow('Calcification Result', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
