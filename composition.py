import cv2 as cv
import numpy as np
import skimage.feature as feature
import matplotlib.pyplot as plt
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths

import argparse
import cv2
import os

import joblib

from typing import Tuple
from calcification.existence import existence

compositionModel = joblib.load("compositionModel.sav")
echogenicityModel = joblib.load("echogenicityModel.sav")

def composition(img: np.ndarray, seg: np.ndarray) -> str:
    maskGlandWithNodule = seg.copy()
    maskGlandWithoutNodule = seg.copy()
    maskNoduleOnly = seg.copy()

    # 0 = Background
    # 1 = Thyroid Gland
    # 2 = Thyroid Nodule

    # Thyroid Gland with Thyroid Nodule
    maskGlandWithNodule[maskGlandWithNodule == 2] = 1

    # Thyroid Gland without Thyroid Nodule
    maskGlandWithoutNodule[maskGlandWithoutNodule != 1] = 0

    # Thyroid Nodule only
    maskNoduleOnly[maskNoduleOnly != 2] = 0

    # Convert marked pixels to white
    maskGlandWithNodule = np.multiply(maskGlandWithNodule, 255)
    maskGlandWithoutNodule = np.multiply(maskGlandWithoutNodule, 255)
    maskNoduleOnly = np.multiply(maskNoduleOnly, 255)

    # Mask the over original image
    resultGlandWithNodule = cv.bitwise_and(img, maskGlandWithNodule)
    resultGlandWithoutNodule = cv.bitwise_and(img, maskGlandWithoutNodule)
    resultNoduleOnly = cv.bitwise_and(img, maskNoduleOnly)

    desc = LocalBinaryPatterns(24, 8)

    # Preprocess images by converting all to grayscale
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # maskNoduleOnly = cv.cvtColor(maskNoduleOnly, cv.COLOR_BGR2GRAY)
    # resultNoduleOnly = cv.cvtColor(resultNoduleOnly, cv.COLOR_BGR2GRAY)
    # maskGlandWithNodule = cv.cvtColor(maskGlandWithNodule, cv.COLOR_BGR2GRAY)
    # resultGlandWithNodule = cv.cvtColor(resultGlandWithNodule, cv.COLOR_BGR2GRAY)

    # Find the contour of the images
    contoursNoduleOnly, _NoduleOnly = cv.findContours(
        maskNoduleOnly, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contoursGlandWithNodule, _GlandWithNodule = cv.findContours(
        maskGlandWithNodule, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    contourNoduleOnly = max(contoursNoduleOnly, key=cv2.contourArea)
    contourGlandWithNodule = max(contoursGlandWithNodule, key=cv2.contourArea)

    # Cropping the image to get rid of most 0 pixel background
    xNO, yNO, wNO, hNO = cv.boundingRect(contourNoduleOnly)
    croppedThyroidNodule = resultNoduleOnly[yNO : yNO + hNO, xNO : xNO + wNO]

    xGWN, yGWN, wGWN, hGWN = cv.boundingRect(contourGlandWithNodule)
    croppedThyroidGland = resultGlandWithNodule[yGWN : yGWN + hGWN, xGWN : xGWN + wGWN]

    # Resizing the ROI to fill the whole square
    resizedThyroidNodule = croppedThyroidNodule.copy()
    kernelThyroidNodule = np.ones((3, 3), np.uint8)
    resizedThyroidNodule = cv.dilate(
        resizedThyroidNodule, kernelThyroidNodule, iterations=3
    )

    resizedThyroidGland = croppedThyroidGland.copy()
    kernelThyroidGland = np.ones((5, 5), np.uint8)
    resizedThyroidGland = cv.dilate(resizedThyroidGland, kernelThyroidGland, iterations=5)

    thyroidNoduleSquare = resizedThyroidNodule
    thyroidGlandSquare = resizedThyroidGland

    # Variable to save Nodule Composition result
    noduleComposition = ""
    # Cystic
    # Solid
    # Mixed

    # Run Nodule Composition SVM Model
    histComposition = desc.describe(thyroidNoduleSquare)
    composition = compositionModel.predict(histComposition.reshape(1, -1))
    noduleComposition = composition[0]
    return noduleComposition

def composition_wit_enchogenicity(img: np.ndarray, seg: np.ndarray) -> Tuple[str, str]:
    maskGlandWithNodule = seg.copy()
    maskGlandWithoutNodule = seg.copy()
    maskNoduleOnly = seg.copy()

    # 0 = Background
    # 1 = Thyroid Gland
    # 2 = Thyroid Nodule

    # Thyroid Gland with Thyroid Nodule
    maskGlandWithNodule[maskGlandWithNodule == 2] = 1

    # Thyroid Gland without Thyroid Nodule
    maskGlandWithoutNodule[maskGlandWithoutNodule != 1] = 0

    # Thyroid Nodule only
    maskNoduleOnly[maskNoduleOnly != 2] = 0

    # Convert marked pixels to white
    maskGlandWithNodule = np.multiply(maskGlandWithNodule, 255)
    maskGlandWithoutNodule = np.multiply(maskGlandWithoutNodule, 255)
    maskNoduleOnly = np.multiply(maskNoduleOnly, 255)

    # Mask the over original image
    resultGlandWithNodule = cv.bitwise_and(img, maskGlandWithNodule)
    resultGlandWithoutNodule = cv.bitwise_and(img, maskGlandWithoutNodule)
    resultNoduleOnly = cv.bitwise_and(img, maskNoduleOnly)

    desc = LocalBinaryPatterns(24, 8)

    # Preprocess images by converting all to grayscale
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # maskNoduleOnly = cv.cvtColor(maskNoduleOnly, cv.COLOR_BGR2GRAY)
    # resultNoduleOnly = cv.cvtColor(resultNoduleOnly, cv.COLOR_BGR2GRAY)
    # maskGlandWithNodule = cv.cvtColor(maskGlandWithNodule, cv.COLOR_BGR2GRAY)
    # resultGlandWithNodule = cv.cvtColor(resultGlandWithNodule, cv.COLOR_BGR2GRAY)

    # Find the contour of the images
    contoursNoduleOnly, _NoduleOnly = cv.findContours(
        maskNoduleOnly, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contoursGlandWithNodule, _GlandWithNodule = cv.findContours(
        maskGlandWithNodule, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    contourNoduleOnly = max(contoursNoduleOnly, key=cv2.contourArea)
    contourGlandWithNodule = max(contoursGlandWithNodule, key=cv2.contourArea)

    # Cropping the image to get rid of most 0 pixel background
    xNO, yNO, wNO, hNO = cv.boundingRect(contourNoduleOnly)
    croppedThyroidNodule = resultNoduleOnly[yNO : yNO + hNO, xNO : xNO + wNO]

    xGWN, yGWN, wGWN, hGWN = cv.boundingRect(contourGlandWithNodule)
    croppedThyroidGland = resultGlandWithNodule[yGWN : yGWN + hGWN, xGWN : xGWN + wGWN]

    # Resizing the ROI to fill the whole square
    resizedThyroidNodule = croppedThyroidNodule.copy()
    kernelThyroidNodule = np.ones((3, 3), np.uint8)
    resizedThyroidNodule = cv.dilate(
        resizedThyroidNodule, kernelThyroidNodule, iterations=3
    )

    resizedThyroidGland = croppedThyroidGland.copy()
    kernelThyroidGland = np.ones((5, 5), np.uint8)
    resizedThyroidGland = cv.dilate(resizedThyroidGland, kernelThyroidGland, iterations=5)

    thyroidNoduleSquare = resizedThyroidNodule
    thyroidGlandSquare = resizedThyroidGland

    # Variable to save Nodule Composition result
    noduleComposition = ""
    # Cystic
    # Solid
    # Mixed

    # Variable to save Nodule Echogenicity result
    noduleEchogenicity = ""
    # Anechoic 无
    # Hyperechoic 高
    # Isoechoic 等
    # Hypoechoic 底
    # Very Hypoechoic 极低

    # Cystic == Anechoic

    # Variable to save Nodule Calcification result
    seg_nodule = seg.copy()
    seg_nodule[seg_nodule < 2] = 0
    nodule = np.minimum(img, seg_nodule // 2 * 255)
    noduleCalcification, _ = existence(nodule, seg)

    # Run Nodule Composition SVM Model
    histComposition = desc.describe(thyroidNoduleSquare)
    composition = compositionModel.predict(histComposition.reshape(1, -1))
    noduleComposition = composition[0]

    histEchogenicity = desc.describe(thyroidGlandSquare)
    if noduleComposition == "Cystic":
        noduleEchogenicity = "Anechoic"
    elif noduleCalcification == True:
        noduleEchogenicity = "Echogenicity cannot be assessed"
    else:
        echogenicity = echogenicityModel.predict(histEchogenicity.reshape(1, -1))
        noduleEchogenicity = echogenicity[0]

    return noduleComposition, noduleEchogenicity
