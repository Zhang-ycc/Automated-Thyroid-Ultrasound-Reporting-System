import cv2
import numpy as np
import os
from PIL import Image
from typing import Optional

def and_8bitwise(img1, img2):
    return np.where(img1 == img2, img1, 0)

if __name__ == '__main__':
    icon: Optional[np.ndarray] = None
    for file in os.listdir('./'):
        if file.startswith('neck02-'):
            img = cv2.imread(os.path.join('./', file))
            icon = img if icon is None else and_8bitwise(icon, img)
    if icon is not None:
        icon = icon.transpose(2, 0, 1)[0]
        print(icon.shape)
        img = Image.fromarray(icon)
        img.save('neck02.png')
