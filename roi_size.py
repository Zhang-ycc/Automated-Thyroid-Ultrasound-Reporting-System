import cv2
import numpy as np
from typing import Tuple

def get_roi_w_h(seg: np.ndarray) -> Tuple[float, float]:
    cnts, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in cnts]
    max_box_s = 0
    max_box = (0, 0)
    for box in bounding_boxes:
        _, _, w, h = box
        if w * h > max_box_s:
            max_box_s = w * h
            max_box = (w, h)
    return max_box
