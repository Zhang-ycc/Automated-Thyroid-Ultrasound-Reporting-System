import math

import cv2
import numpy as np
from typing import Tuple


def compute_curvature(points):
    x, y = points[:, 0], points[:, 1]
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs((dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt ** 2 + dy_dt ** 2) ** 1.5)
    return curvature


def preprocess(mask):
    # 将0和1转换为0，将2转换为255
    mask = np.uint8(np.where(mask == 2, 255, 0))
    # 边缘检测
    edges = cv2.Canny(mask, 30, 150)
    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 提取最外圈边缘
    outer_contour = max(contours, key=cv2.contourArea)

    return outer_contour


def is_clear(image, mask) -> bool:
    """
    :param image: 超声原图
    :param mask: 掩码矩阵
    :return: True:清晰; False:不清晰
    """
    # 提取最外圈边缘
    outer_contour = preprocess(mask)

    # 创建空白图像
    height, width = image.shape[:2]
    contour_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 绘制最外面的轮廓
    cv2.drawContours(contour_image, [outer_contour], -1, (0, 255, 0), 2)

    kernel_size = int(30 * math.sqrt(cv2.arcLength(outer_contour, True) / 1000))

    # 对轮廓进行膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    expanded_contour = cv2.dilate(contour_image, kernel)
    expanded_contour = cv2.cvtColor(expanded_contour, cv2.COLOR_BGR2GRAY)

    # 将边缘应用原图上
    result = cv2.bitwise_and(image, image, mask=expanded_contour)

    # 应用阈值处理
    _, threshold = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)

    # 轮廓检测
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 获取最大轮廓的边界框
    x, y, width, height = cv2.boundingRect(max_contour)

    # 提取ROI区域
    roi = result[y:y + height, x:x + width]

    resized_roi = cv2.resize(roi, image.shape)

    # 显示ROI区域

    # 设置阈值，根据梯度统计特征判断边缘清晰度
    threshold = 2  # 平均梯度阈值

    # 计算图像的梯度
    gradient_x = cv2.Sobel(resized_roi, cv2.CV_64F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(resized_roi, cv2.CV_64F, 0, 1, ksize=1)
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # 计算梯度均值
    mean_gradient = np.mean(gradient)

    print(mean_gradient)

    return mean_gradient > threshold


def is_smooth(mask) -> bool:
    """
    :param mask: 掩码矩阵
    :return: True:光滑; False:不光滑
    """
    # 提取最外圈边缘
    outer_contour = preprocess(mask)

    # 计算凸包
    convex_hull = cv2.convexHull(outer_contour)

    origin_area = cv2.contourArea(outer_contour)
    hull_area = cv2.contourArea(convex_hull)

    threshold_smooth = 0.04

    difference = abs(hull_area - origin_area) / origin_area

    return difference < threshold_smooth


def edge_detail(image, mask, props):

    detail_image = np.stack((image,) * 3, axis=-1)

    # 提取最外圈边缘
    outer_contour = preprocess(mask)
    # 计算凸包
    hull = cv2.convexHull(outer_contour)
    # 计算凸缺陷
    defects = cv2.convexityDefects(outer_contour, cv2.convexHull(outer_contour, returnPoints=False))

    cv2.drawContours(detail_image, [outer_contour], -1, (0, 255, 0), 2)
    cv2.drawContours(detail_image, [hull], -1, (0, 0, 255), 2)

    if is_smooth(mask):
        return "光滑", detail_image

    total_depth = 0
    max_angle = 0
    max_depth = 0
    acute = False
    # 根据凸缺陷判断边缘类型
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            start = tuple(outer_contour[s][0])
            end = tuple(outer_contour[e][0])
            far = tuple(outer_contour[f][0])
            cv2.line(detail_image, start, end, (255, 0, 0), 2)
            cv2.circle(detail_image, far, 5, (0, 255, 0), -1)

            # 计算凸缺陷的深度
            depth = defects[i, 0, 3] / 256.0

            # 计算边缘的两条向量
            vector1 = np.array(start) - np.array(far)
            vector2 = np.array(end) - np.array(far)
            # 计算两个向量的夹角（弧度）
            angle = np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
            angle = angle * 180 / np.pi
            if angle < 90:
                practical_depth = (props['delta_x'] + props['delta_y']) / 2 * depth * 10
                if practical_depth > 0.5:
                    acute = True

            if depth > max_depth:
                max_depth = depth
                max_angle = angle

            total_depth += depth

    if acute:
        return "成角", detail_image

    mean_depth = total_depth / defects.shape[0]
    normalize_depth = mean_depth / cv2.arcLength(outer_contour, True)

    if max_angle > 90:
        return "分叶", detail_image
    elif normalize_depth > 5e-3:
        return "毛刺", detail_image
    else:
        return "毛糙", detail_image


def edge_process(image, mask) -> Tuple[bool, bool, np.ndarray]:
    """
    :param image: 超声原图
    :param mask: 掩码矩阵
    :return: 是否清晰、是否光滑、轮廓标注
    """
    outer_contour = preprocess(mask)
    contour_image = np.stack((image,) * 3, axis=-1)
    cv2.drawContours(contour_image, [outer_contour], -1, (0, 255, 0), 2)
    print("[DEBUG]", contour_image.shape)
    return is_clear(image, mask), is_smooth(mask), contour_image


if __name__ == '__main__':  # 读取图像
    img = cv2.imread('cases/imagesTr/SJTU_041_0000.png', 0)
    msk = cv2.imread('cases/labelsTr/SJTU_041.png', 0)
    print(edge_process(img, msk))
