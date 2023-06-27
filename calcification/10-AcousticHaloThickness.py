import cv2 as cv
import numpy as np

if __name__ == '__main__':
    mask = cv.imread('example/Dataset002_SJTUThyroid/labelsTr01/SJTU_005.png', 0)  # 读取灰度图像
    mask = mask * 255

    image = cv.imread('example/Dataset002_SJTUThyroid/imagesTr/SJTU_005_0000.png')  # 读取灰度图像
    height, width = image.shape[:2]
    # 将掩膜应用于原始图像，提取椭圆内的图像
    IMAGE = cv.bitwise_and(image, image, mask=mask)

    IMAGE = cv.cvtColor(IMAGE, cv.COLOR_BGR2GRAY)

    IMAGE = cv.GaussianBlur(IMAGE, (5, 5), 0)  # 高斯平滑滤波

    edge = cv.imread('example/Dataset002_SJTUThyroid/labelsTr02/SJTU_005.png', 0)  # 读取灰度图像

    _, threshold = cv.threshold(edge, 127, 255, cv.THRESH_BINARY)
    nodules_contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    edgePoints = nodules_contours[0]

    M = cv.moments(edgePoints)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    print(center_x, center_y)

    thickness_values = []

    for point in edgePoints:
        x, y = point[0]
        print(x, y)
        thickness = 0
        x0 = x
        y0 = y

        while True:
            if 0 < IMAGE[y, x] < 30:
                thickness += 1
            else:
                print(thickness)
                thickness_values.append(thickness)
                break

            # Calculate the direction vector
            dx = x - center_x
            dy = y - center_y

            # Update the position (x, y) along the line
            if abs(dx) >= abs(dy):
                x += 1 if dx > 0 else -1
                y += int(round(dy / abs(dx))) if dx != 0 else 0
            else:
                x += int(round(dx / abs(dy))) if dy != 0 else 0
                y += 1 if dy > 0 else -1

            if x < 0 or x >= edge.shape[1] or y < 0 or y >= edge.shape[0]:
                break

        cv.line(IMAGE, (x0, y0), (x, y), (255, 255, 255), 1)

    non_zero_thickness_values = np.array(thickness_values)[np.array(thickness_values) != 0]

    # 计算平均厚度
    average_thickness = np.mean(non_zero_thickness_values)
    # average_thickness = np.mean(thickness_values)
    print(average_thickness)

    # Display the IMAGE with the extended range marked
    cv.imshow('Extended Range', IMAGE)
    cv.waitKey(0)
    cv.destroyAllWindows()
