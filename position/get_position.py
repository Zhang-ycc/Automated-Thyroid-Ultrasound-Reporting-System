import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread
from typing import Tuple
try:
    from .get_organ import is_trachea
except:
    from get_organ import is_trachea

template1_filename = './position/neck01.png'
template2_filename = './position/neck02.png'

def get_neck_position_from_img(img: np.ndarray) -> Tuple[float, np.ndarray]:
    template1 = cv2.imread(template1_filename).transpose(2, 0, 1)[0]
    template2 = cv2.imread(template2_filename).transpose(2, 0, 1)[0]
    result1 = cv2.matchTemplate(img, template1, cv2.TM_CCOEFF_NORMED)
    result2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
    _, max_val1, _, max_loc1 = cv2.minMaxLoc(result1)
    _, max_val2, _, max_loc2 = cv2.minMaxLoc(result2)
    if max_val1 >= max_val2:
        x, y = max_loc1
        w, h = template1.shape
    else:
        x, y = max_loc2
        w, h = template2.shape
    return max(max_val1, max_val2), img[y:y+h, x:x+w]

def get_scan_position_from_neck(neck: np.ndarray) -> np.ndarray:
    scan = neck.copy()
    _, scan = cv2.threshold(scan, 230, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    scan = cv2.morphologyEx(scan, cv2.MORPH_CLOSE, kernel, iterations=5)
    return scan

def get_scan_direction_from_scan(scan: np.ndarray) -> str:
    contours, _ = cv2.findContours(scan, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (h, w), r = cv2.minAreaRect(contours[0])
    print(h, w, r)
    if abs(r - 90) > 5:
        return f'斜扫（{r:.2f}度）'
    elif h > w:
        return '竖扫'
    else:
        return '横扫'

def get_position(img: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, str]:
    val, neck = get_neck_position_from_img(img)
    if val < 0.3:
        exist, convex = is_trachea(img, seg)
        return convex, '横扫' if exist else '竖扫'
    scan = get_scan_position_from_neck(neck)
    dir = get_scan_direction_from_scan(scan)
    return neck, dir

if __name__ == '__main__':
    template1_filename = './neck01.png'
    template2_filename = './neck02.png'
    # filename = '../cases/202303080911100016SMP.DCM' # class 1 horizontal
    # filename = '../cases/IM_0027' # class 2 vertical
    # filename = '../cases/202303080913520027SMP.DCM' # class 1 vertical
    # filename = '../cases/202303081019410072SMP.DCM' # class 1 tilt
    # filename = '../cases/IM_0011' # class 2 no scan
    # filename = '../cases/IM_0025' # class 2 no scan
    filename = '../cases/IM_0203' # class 2 no scan with trachea
    ds = dcmread(filename)
    array = ds.pixel_array
    if len(array.shape) == 3:
        array = array.transpose(2, 0, 1)[0]
    val, _ = get_neck_position_from_img(array)
    print(val)
