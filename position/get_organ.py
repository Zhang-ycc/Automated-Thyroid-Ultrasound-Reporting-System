import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def get_us_mask_from_img(img: np.ndarray) -> np.ndarray:
    us = img.copy()
    us[us >= 5] = 255
    us[us < 5] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(us)
    max_s_idx = 1
    for idx in range(2, num_labels):
        if stats[idx][4] > stats[max_s_idx][4]:
            max_s_idx = idx
    labels[labels != max_s_idx] = 0
    labels[labels == max_s_idx] = 255
    us = np.minimum(us, labels).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    us = cv2.morphologyEx(us, cv2.MORPH_CLOSE, kernel, iterations=1)
    us = cv2.erode(us, kernel, iterations=10)
    return us

def get_high_light_from_img(img: np.ndarray) -> np.ndarray:
    high_light = img.copy()
    _, high_light = cv2.threshold(high_light, 160, 255, cv2.THRESH_BINARY)
    mask = get_us_mask_from_img(img)
    high_light = np.minimum(high_light, mask)
    return high_light

def get_max_box_from_img(img: np.ndarray) -> Tuple[int, int, int, int]:
    cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in cnts]
    max_box_s = 0
    max_box = (0, 0, 0, 0)
    for box in bounding_boxes:
        _, _, w, h = box
        if w * h > max_box_s:
            max_box_s = w * h
            max_box = box
    return max_box

def get_max_convex_defect_from_img(img: np.ndarray) -> Tuple[int, np.ndarray]:
    debug_img = img.copy()
    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
    cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug_img, cnts, -1, (0, 225, 0), 3)
    max_defect = 0
    for cnt in cnts:
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None:
            max_defect = max(max_defect, defects[:, :, 3].max() / 256)
            for j in range(defects.shape[0]):
                s, e, f, d = defects[j, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(debug_img, start, end, (0, 0, 225), 2)
                cv2.circle(debug_img, far, 5, (225, 0, 0), -1)
    return max_defect, debug_img

def is_trachea(img: np.ndarray, seg: np.ndarray) -> Tuple[bool, np.ndarray]:
    TRACHEA_THRESHOLD = 0.5
    thyroid_mask = seg.copy()
    thyroid_mask[thyroid_mask > 0] = 255
    max_defect, convex = get_max_convex_defect_from_img(thyroid_mask)
    max_box_h = get_max_box_from_img(thyroid_mask)[3]
    return max_defect / max_box_h > TRACHEA_THRESHOLD, convex

def is_skin(img: np.ndarray, seg: np.ndarray) -> bool:
    SKIN_THRESHOLD = 100
    high_light = get_high_light_from_img(img)
    thyroid_mask = seg.copy()
    thyroid_mask[thyroid_mask > 0] = 255
    max_box = get_max_box_from_img(thyroid_mask)
    thyroid_mask = 255 - thyroid_mask
    high_light = np.minimum(high_light, thyroid_mask)
    return high_light[:max_box[1], :].sum() > SKIN_THRESHOLD

def get_organ(img: np.ndarray, seg: np.ndarray) -> Tuple[bool, bool]:
    return is_skin(img, seg), is_trachea(img, seg)[0]

if __name__ == '__main__':
    img_idx = '047'
    img_filename = f'../cases/Dataset002_SJTUThyroid/imagesTr/SJTU_{img_idx}_0000.png'
    seg_filename = f'../cases/Dataset002_SJTUThyroid/labelsTr/SJTU_{img_idx}.png'
    img = cv2.imread(img_filename)
    img = img.transpose(2, 0, 1)[0]
    seg = cv2.imread(seg_filename)
    seg = seg.transpose(2, 0, 1)[0]
    print(get_organ(img, seg)[:2])
