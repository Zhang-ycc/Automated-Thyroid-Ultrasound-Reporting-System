import cv2 as cv
import numpy as np
import skimage.feature as feature
import matplotlib.pyplot as plt
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths

import argparse
import cv2
import os

import joblib

# Define paths of the image files
filePathOriginal = r"D:\SJTU\2022-2023(6)\VisualComputing\FinalProject\Dataset002_SJTUThyroid\Dataset002_SJTUThyroid\imagesTr\SJTU_022_0000.png"
filePathMask = r"D:\SJTU\2022-2023(6)\VisualComputing\FinalProject\Dataset002_SJTUThyroid\Dataset002_SJTUThyroid\labelsTr\SJTU_022.png"

originalImage = cv.imread(filePathOriginal)
maskImage = cv.imread(filePathMask)

maskGlandWithNodule = maskImage.copy()
maskGlandWithoutNodule = maskImage.copy()
maskNoduleOnly = maskImage.copy()

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
resultGlandWithNodule = cv.bitwise_and(originalImage, maskGlandWithNodule)
resultGlandWithoutNodule = cv.bitwise_and(originalImage, maskGlandWithoutNodule)
resultNoduleOnly = cv.bitwise_and(originalImage, maskNoduleOnly)

desc = LocalBinaryPatterns(24, 8)

# Preprocess images by converting all to grayscale
originalImage = cv.cvtColor(originalImage, cv.COLOR_BGR2GRAY)
maskNoduleOnly = cv.cvtColor(maskNoduleOnly, cv.COLOR_BGR2GRAY)
resultNoduleOnly = cv.cvtColor(resultNoduleOnly, cv.COLOR_BGR2GRAY)
maskGlandWithNodule = cv.cvtColor(maskGlandWithNodule, cv.COLOR_BGR2GRAY)
resultGlandWithNodule = cv.cvtColor(resultGlandWithNodule, cv.COLOR_BGR2GRAY)

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
noduleCalcification = False

# Run Nodule Composition SVM Model
compositionModel = joblib.load("compositionModel.sav")
histComposition = desc.describe(thyroidNoduleSquare)
composition = compositionModel.predict(histComposition.reshape(1, -1))
noduleComposition = composition[0]
print(noduleComposition)

# Run Nodule Echogenicity SVM Model
echogenicityModel = joblib.load("echogenicityModel.sav")
histEchogenicity = desc.describe(thyroidGlandSquare)
if noduleComposition == "Cystic":
    noduleEchogenicity = "Anechoic 无"
elif noduleCalcification == True:
    noduleEchogenicity = "Echogenicity cannot be assessed"
else:
    echogenicity = echogenicityModel.predict(histEchogenicity.reshape(1, -1))
    noduleEchogenicity = echogenicity[0]

print(noduleEchogenicity)
