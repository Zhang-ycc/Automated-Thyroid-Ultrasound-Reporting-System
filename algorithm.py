import numpy as np
from abc import ABC, abstractmethod
from composition import composition, composition_wit_enchogenicity
from edge import edge_process, edge_detail
from calcification.existence import existence
from calcification.calcification_position import CalcificationPosition
from calcification.clacification import clacification
from calcification.AcousticHaloThickness import AcousticHaloThick
from position.get_position import get_position
from position.get_organ import get_organ
from roi_size import get_roi_w_h
from typing import Optional, Tuple
from utils.utils import *

class Algorithm(ABC):
    @staticmethod
    @abstractmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        return np.zeros_like(img), ''

class SegmentationAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        seg_thyroid = seg.copy()
        seg_thyroid[seg_thyroid > 0] = 1
        seg_nodule = seg.copy()
        seg_nodule[seg_nodule < 2] = 0
        desc = '甲状腺部分有结节'
        if seg_nodule.max() < 2:
            desc = '甲状腺部分无结节'
        show_seg = np.minimum(img, seg_thyroid * 255)
        show_seg = np.maximum(show_seg, seg_nodule // 2 * 255)
        return show_seg, desc, seg_nodule.max() == 2

class ROISizeAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        seg_nodule = seg.copy()
        seg_nodule[seg_nodule < 2] = 0
        seg_nodule[seg_nodule > 0] = 255
        nodule = np.minimum(img, seg_nodule)
        w, h = get_roi_w_h(seg_nodule)
        w *= props['delta_x'] * 10
        h *= props['delta_y'] * 10
        s = seg_nodule.sum() // 255 * props['delta_x'] * 10 * props['delta_y'] * 10
        _, position = get_position(img, seg)
        if position == '横扫':
            return nodule, f'结节的大小为{s:.4f}mm2，左右径为{w:.4f}mm，前后径为{h:.4f}mm', (0, w, h)
        else:
            return nodule, f'结节的大小为{s:.4f}mm2，上下径为{w:.4f}mm，前后径为{h:.4f}mm', (w, 0, h)

class EdgeAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        if seg.max() < 2:
            return np.ones_like(img), '未找到结节', None
        else:
            clear, smooth, processed_img = edge_process(img, seg)
            return processed_img, f'结节{boolean_to_zh_cn_is(clear)}清晰的，结节{boolean_to_zh_cn_is(smooth)}光滑的', (clear, smooth)

class NoduleInnerMassAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        seg_nodule = seg.copy()
        seg_nodule[seg_nodule < 2] = 0
        nodule = np.minimum(img, seg_nodule // 2 * 255)
        comp = composition(img, seg)
        desc = f'结节内质的基本性质为：{en_to_zh_cn_composition(comp)}'
        return nodule, desc, en_to_zh_cn_composition(comp)

class NoduleCalcificationAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        seg_nodule = seg.copy()
        seg_nodule[seg_nodule < 2] = 0
        nodule = np.minimum(img, seg_nodule // 2 * 255)
        exist, processed_img = existence(nodule, seg)
        comp = composition(img, seg)
        if exist and comp == 'Cystic':
            exist = 2
        return processed_img, exist_to_calcification(exist), None

class ThyroidCalcificationAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        seg_thyroid = seg.copy()
        seg_thyroid[seg_thyroid > 1] = 1
        nodule = np.minimum(img, seg_thyroid * 255)
        processed_img = CalcificationPosition(seg, nodule)
        return processed_img, '', None

class PositionAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        processed, position = get_position(img, seg)
        is_skin, is_trachea = get_organ(img, seg)
        organ_desc = f'甲状腺附近{boolean_to_zh_cn_exist(is_skin)}皮肤，{boolean_to_zh_cn_exist(is_trachea)}气管。'
        return processed, f'{position}超声可见，{organ_desc}', None

class NoduleShapeAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        if seg.max() < 2:
            return np.ones_like(img), '未找到结节', None
        else:
            detail, processed_img = edge_detail(img, seg, props)
            return processed_img, f'结节边缘{detail}', None


class HaloAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        mask = seg.copy()
        mask[mask > 2] = 0
        edge = seg.copy()
        edge[edge < 2] = 0
        edge = edge // 2 * 255
        thickness = AcousticHaloThick(img, mask, edge) * (props['delta_x'] + props['delta_y']) * 5
        nodule = np.minimum(img, edge)
        if thickness >= 1e-4:
            return nodule, f'声晕的厚度为{thickness:.4f}mm', None
        else:
            return nodule, f'无声晕', None

class InnerMassClassificationAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        seg_nodule = seg.copy()
        seg_nodule[seg_nodule < 2] = 0
        nodule = np.minimum(img, seg_nodule // 2 * 255)
        comp, echo = composition_wit_enchogenicity(img, seg)
        desc = f'结节内质的基本性质为：{en_to_zh_cn_echogenicity(echo)}（{en_to_zh_cn_composition(comp)}）'
        return nodule, desc, en_to_zh_cn_echogenicity(echo)

class CalcificationClassificationAlgorithm(Algorithm):
    @staticmethod
    def get_image_and_description(
        img: np.ndarray,
        seg: np.ndarray,
        props: dict
    ) -> Tuple[np.ndarray, str, any]:
        mask = seg.copy()
        mask[mask > 2] = 0
        edge = seg.copy()
        edge[edge < 2] = 0
        edge = edge // 2 * 255
        shape_types, result = clacification(img, mask, edge)
        if len(shape_types):
            desc = f'甲状腺中共有{len(shape_types)}种钙化，包括：{"、".join(list(shape_types))}'
        else:
            desc = '甲状腺中没有出现钙化'
        is_micro = False
        for shape_type in shape_types:
            if desc.find('点') != -1 or desc.find('微') != -1:
                is_micro = True
        return result, desc, is_micro
