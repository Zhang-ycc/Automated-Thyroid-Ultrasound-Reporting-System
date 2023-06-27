import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def getJ(image, mask):
    name = image
    image = cv.imread('example/Dataset002_SJTUThyroid/imagesTr/' + image)
    mask = cv.imread('example/Dataset002_SJTUThyroid/labelsTr02/' + mask, 0)
    Image = cv.bitwise_and(image, image, mask=mask)

    # 显示结果图像
    cv.imwrite(f'nodules/{name}', Image)
    # cv.imshow('Result', Image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

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



if __name__ == '__main__':
    names = get_names('example/Dataset002_SJTUThyroid/imagesTr')
    maskNames = get_names('example/Dataset002_SJTUThyroid/labelsTr02')
    for name, mask_name in zip(names, maskNames):
        print(name, mask_name)
        getJ(name, mask_name)
