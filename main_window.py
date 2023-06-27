from algorithm import *
from image_widget import ImageWidget
from MainWindow_ui import Ui_MainWindow
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFileDialog, QMainWindow, QWidget
from PySide6.QtGui import QAction
from typing import List, Optional, Tuple, Type
from typing import Optional, Tuple, Type

from report.generate import Report
from utils.utils import *


class MainWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.all_functional_actions: List[QAction] = [
            self.ui.action_segmentation,
            self.ui.action_roi_size,
            self.ui.action_edge,
            self.ui.action_nodule_inner_mass,
            self.ui.action_nodule_calcification,
            self.ui.action_thyroid_calcification,
            self.ui.action_position,
            self.ui.action_nodule_shape,
            self.ui.action_halo,
            self.ui.action_inner_mass_classification,
            self.ui.action_calcification_classification,
            self.ui.action_export_report
        ]
        self.implemented_functional_actions: List[QAction] = [
            self.ui.action_segmentation,
            self.ui.action_roi_size,
            self.ui.action_edge,
            self.ui.action_nodule_inner_mass,
            self.ui.action_nodule_calcification,
            self.ui.action_thyroid_calcification,
            self.ui.action_position,
            self.ui.action_nodule_shape,
            self.ui.action_halo,
            self.ui.action_inner_mass_classification,
            self.ui.action_calcification_classification,
            self.ui.action_export_report
        ]
        self.action_to_algorithm: List[Tuple[QAction, Type[Algorithm], QImage.Format]] = [
            (self.ui.action_segmentation, SegmentationAlgorithm, QImage.Format.Format_Grayscale8),
            (self.ui.action_roi_size, ROISizeAlgorithm, QImage.Format.Format_Grayscale8),
            (self.ui.action_edge, EdgeAlgorithm, QImage.Format.Format_RGB888),
            (self.ui.action_nodule_inner_mass, NoduleInnerMassAlgorithm, QImage.Format.Format_Grayscale8),
            (self.ui.action_nodule_calcification, NoduleCalcificationAlgorithm, QImage.Format.Format_RGB888),
            (self.ui.action_thyroid_calcification, ThyroidCalcificationAlgorithm, QImage.Format.Format_RGB888),
            (self.ui.action_position, PositionAlgorithm, QImage.Format.Format_Grayscale8),
            (self.ui.action_nodule_shape, NoduleShapeAlgorithm, QImage.Format.Format_RGB888),
            (self.ui.action_halo, HaloAlgorithm, QImage.Format.Format_Grayscale8),
            (self.ui.action_inner_mass_classification, InnerMassClassificationAlgorithm,
             QImage.Format.Format_Grayscale8),
            (self.ui.action_calcification_classification, CalcificationClassificationAlgorithm,
             QImage.Format.Format_RGB888)
        ]
        self.ui.action_open_file.triggered.connect(self.on_click_open_file)
        self.ui.action_close_file.triggered.connect(self.on_click_close_file)
        self.ui.tabWidget.tabCloseRequested.connect(self.on_click_close_tab)
        self.ui.tabWidget.currentChanged.connect(self.on_change_tab)
        self.ui.action_export_report.triggered.connect(self.on_click_export_report)

        for action, algorithm, format in self.action_to_algorithm:
            action.triggered.connect(self.on_click_function_generator(algorithm, format))
        for action in self.all_functional_actions:
            action.setEnabled(False)

    def enable_functional_actions(self, enable: bool) -> None:
        for button in self.implemented_functional_actions:
            button.setEnabled(enable)

    def get_current_image_widget(self) -> Optional[ImageWidget]:
        if self.ui.tabWidget.count() <= 0:
            return None
        return self.ui.tabWidget.currentWidget()  # type: ignore

    def display_processed_image(self, img: np.ndarray, format: QImage.Format = QImage.Format.Format_Grayscale8) -> None:
        image_widget = self.get_current_image_widget()
        if image_widget != None:
            image_widget.display_processed_image(img, format)

    def display_text_message(self, message: str) -> None:
        self.ui.statusbar.showMessage(message)

    def get_current_img_seg(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
        image_widget = self.get_current_image_widget()
        return (None, None, None) if image_widget is None else image_widget.get_image_segmentation_and_props()

    def is_current_seg_done(self) -> bool:
        image_widget = self.get_current_image_widget()
        return image_widget is not None and image_widget.is_segmentation_done()

    @Slot()
    def on_click_open_file(self) -> None:
        dicom_filename, _ = QFileDialog.getOpenFileName(self, '打开文件', './', 'DICOM Files (*.dcm)')
        if dicom_filename != '':
            ds, props = read_dicom_file(dicom_filename)
            pix_arr = dicom_file_dataset_to_ndarray(ds)
            image_widget = ImageWidget(pix_arr, props, self)
            image_widget.get_segmentation_signal.connect(self.tab_get_segmentation_done)
            new_idx = self.ui.tabWidget.addTab(image_widget, dicom_filename.split('/')[-1] + '（未分割）')
            self.ui.tabWidget.setCurrentIndex(new_idx)
            image_widget.display_original_image()
            self.enable_functional_actions(False)
            self.ui.statusbar.showMessage('分割中...')

    @Slot()
    def on_click_close_file(self) -> None:
        self.ui.tabWidget.removeTab(self.ui.tabWidget.currentIndex())
        if self.ui.tabWidget.count() > 0:
            self.ui.tabWidget.setCurrentIndex(self.ui.tabWidget.count() - 1)
        else:
            self.enable_functional_actions(False)

    @Slot()
    def on_click_export_report(self) -> None:
        img, seg, props = self.get_current_img_seg()
        if img is None or seg is None or props is None:
            return
        report_path, _ = QFileDialog.getSaveFileName(self, '导出报告', './', "报告文档 (*.pdf)")
        if report_path:
            report = Report(img, seg, props)
            report.create_pdf(report_path)

    @Slot()
    def on_click_close_tab(self, index: int) -> None:
        self.ui.statusbar.clearMessage()
        self.ui.tabWidget.removeTab(index)
        if self.ui.tabWidget.count() <= 0:
            self.enable_functional_actions(False)

    @Slot()
    def on_change_tab(self, index: int) -> None:
        if index == -1:
            self.enable_functional_actions(False)
        elif self.is_current_seg_done():
            self.enable_functional_actions(True)
            self.ui.statusbar.showMessage('分割完成')
        else:
            self.enable_functional_actions(False)
            self.ui.statusbar.showMessage('分割中...')

    def on_click_function_generator(self, algorithm: Type[Algorithm],
                                    format: QImage.Format = QImage.Format.Format_Grayscale8):
        @Slot()
        def on_click_function() -> None:
            img, seg, props = self.get_current_img_seg()
            if img is None or seg is None or props is None:
                return
            processed_img, desc, _ = algorithm.get_image_and_description(img, seg, props)
            self.display_processed_image(processed_img, format)
            self.display_text_message(desc)

        return on_click_function

    @Slot()
    def tab_get_segmentation_done(self, widget: QWidget) -> None:
        idx = self.ui.tabWidget.indexOf(widget)
        title = self.ui.tabWidget.tabText(idx)
        self.ui.tabWidget.setTabText(idx, title.replace('（未分割）', '（已分割）'))
        if idx == self.ui.tabWidget.currentIndex():
            self.enable_functional_actions(True)
            self.ui.statusbar.showMessage('分割完成')
