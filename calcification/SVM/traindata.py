import os

import cv2
import numpy as np
import cv2 as cv

# 定义存储特征和标签的列表
X_train = []
y_train = []


def get_names(path):
    names = []
    for root, dirnames, filenames in os.walk(path):
        filenames = sorted(filenames)  # 按照字母或数字顺序排序文件名
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if filename != '.DS_Store':
                names.append(filename)
            # if ext in ['.DCM']:
            #     names.append(filename)

    return names


image_paths = "data/TrainImages"
names = get_names(image_paths)

thresholds = [86, 115, 90, 80, 80, 80, 120, 115, 100, 90,
              100, 85, 85, 120, 90, 100, 120, 140, 100, 80, 80,
              90, 100, 90, 100, 70, 90, 100, 80, 120,
              80, 120, 150, 120, 80, 90, 80, 100, 120, 110,
              110, 110, 100, 70, 90, 90, 90, 80, 80, 100]

# thresholds = [80, 110, 100, 90, 100, 80, 80, 90, 90,
#               100, 120, 130, 110, 110, 70, 100, 100, 90,
#               110, 80, 80, 100, 100, 140]

i = 0
# 遍历每张图像
for name in names:
    # 读取图像
    print(name)
    image = cv2.imread(image_paths+'/'+name)

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 提取特征和标签
    features = gray_image.flatten()
    threshold = thresholds[i]

    # 将特征和标签添加到列表中
    X_train.append(features)
    y_train.append(threshold)
    i += 1

# 转换列表为 NumPy 数组
print(X_train)
# 检查数组的形状
shapes = set(arr.shape for arr in X_train)
print(shapes)

if len(shapes) > 1:
    X_train = [np.resize(arr, (786432,)) for arr in X_train]

X_train = np.array(X_train)
y_train = np.array(y_train)

# 存储特征和标签
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
