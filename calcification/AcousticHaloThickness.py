import cv2 as cv
import numpy as np

def AcousticHaloThick(image, mask, edge):
    mask = mask * 255
    IMAGE = cv.bitwise_and(image, image, mask=mask)
    # IMAGE = cv.cvtColor(IMAGE, cv.COLOR_BGR2GRAY)
    IMAGE = cv.GaussianBlur(IMAGE, (5, 5), 0)

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
        # print(x, y)
        thickness = 0

        while True:
            if IMAGE[y, x] < 50:
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

            # 检查像素位置是否超出边缘图像的范围
            if x < 0 or x >= edge.shape[1] or y < 0 or y >= edge.shape[0]:
                break

    non_zero_thickness_values = np.array(thickness_values)[np.array(thickness_values) != 0]

    # 计算平均厚度
    if len(non_zero_thickness_values) != 0:
        average_thickness = np.mean(non_zero_thickness_values)
    else:
        average_thickness = 0
    return average_thickness
