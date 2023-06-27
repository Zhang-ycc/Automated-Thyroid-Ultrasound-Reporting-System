import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

def ndarray_to_qpixmap(arr: np.ndarray, format: QImage.Format = QImage.Format.Format_Grayscale8) -> QPixmap:
    height, width = arr.shape[:2]
    if format == QImage.Format.Format_Grayscale8:
        bytes_per_line = width
    elif format == QImage.Format.Format_RGB888:
        bytes_per_line = width * 3
    else:
        raise NotImplementedError
    return QPixmap(QImage(arr.data, width, height, bytes_per_line, format))

def set_ndarray_on_qlabel(arr: np.ndarray, label: QLabel, format: QImage.Format = QImage.Format.Format_Grayscale8) -> None:
    H, W = arr.shape[:2]
    max_H = label.height()
    max_W = label.width()
    scale = min(max_H / H, max_W / W)
    rescale_arr = cv2.resize(arr, (int(W * scale), int(H * scale)), cv2.INTER_CUBIC)
    pixmap = ndarray_to_qpixmap(rescale_arr, format)
    label.setPixmap(pixmap)
