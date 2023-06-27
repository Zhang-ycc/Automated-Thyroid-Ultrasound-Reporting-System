import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 读取图像和掩膜
    image = cv.imread('example/photos/26.png')
    mask = cv.imread('example/segs/26.png', 0)

    # 确保掩膜与图像的数据类型和尺寸匹配
    mask = mask.astype(np.uint8)
    mask = cv.resize(mask, (image.shape[1], image.shape[0]))

    # 将掩膜应用于图像
    result = cv.bitwise_and(image, image, mask=mask)

    # 显示结果图像
    cv.imwrite('example/thyroid/26.png', result)
    cv.imshow('Result', result)
    cv.waitKey(0)

    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off'), plt.title(f'$input\_image$')
    plt.subplot(132), plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    plt.axis('off'), plt.title(f'$Seg\_image$')
    plt.subplot(133), plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    plt.axis('off'), plt.title(f'$Result\_image$')
    plt.tight_layout()
    # plt.savefig('result-rg-h/21.png')
    plt.show()