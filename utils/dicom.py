import os
import sys

sys.path.append('..')

import numpy as np
from nnUNet.predictor import Predictor
from pydicom import dcmread
from pydicom.dataset import FileDataset
from typing import Tuple

predictor = Predictor(os.path.abspath('nnUNet/nnUNet_results/Dataset001_SJTUThyroidNodule'
                                      '/nnUNetTrainer__nnUNetPlans__2d'))


def read_dicom_file(filename: str) -> Tuple[FileDataset, dict]:
    ds = dcmread(filename)
    seq = ds[0x0018, 0x6011]
    return ds, {'delta_x': seq[0][0x0018, 0x602c].value, 'delta_y': seq[0][0x0018, 0x602e].value}


def dicom_file_dataset_to_ndarray(ds: FileDataset) -> np.ndarray:
    array = ds.pixel_array
    print(len(array.shape), array.shape)
    if len(array.shape) == 3:
        array = array.transpose(2, 0, 1)[0]
    return array


def get_segmentation(img: np.ndarray) -> np.ndarray:
    seg = predictor.get_segmentation(img[None, None].astype(np.float32))
    return seg[0]


if __name__ == '__main__':
    # filename = '../cases/202303080911100016SMP.DCM' # class 1 horizontal
    # filename = '../cases/IM_0027' # class 2 horizontal
    filename = '../cases/202303080913520027SMP.DCM'  # class 1 vertical
    ds, props = read_dicom_file(filename)
    array = ds.pixel_array
    seq = ds[0x0018, 0x6011]
    print(seq)
    print(seq[0][0x0018, 0x602c].value, seq[0][0x0018, 0x602e].value)
