from ImageWidget_ui import Ui_ImageWidget
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QWidget
from typing import Optional, Tuple
from utils.utils import *

class SegmentationSignal(QObject):
    result = Signal(np.ndarray)

class SegmentationWorker(QRunnable):
    def __init__(self, image: np.ndarray) -> None:
        super().__init__()
        self.img = image
        self.signal = SegmentationSignal()

    @Slot()
    def run(self) -> None:
        seg = get_segmentation(self.img)
        self.signal.result.emit(seg)

class ImageWidget(QWidget):
    get_segmentation_signal = Signal(QWidget)

    def __init__(self, image: np.ndarray, props: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ui = Ui_ImageWidget()
        self.ui.setupUi(self)
        self.img = image
        self.props = props
        self.seg: Optional[np.ndarray] = None
        self.processed: Optional[np.ndarray] = None
        self.processed_format: Optional[QImage.Format] = None

        self.threadpool = QThreadPool()
        worker = SegmentationWorker(image)
        worker.signal.result.connect(self.get_segmentation_result)
        self.threadpool.start(worker)

    def get_image_segmentation_and_props(self) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        return self.img, self.seg, self.props

    def display_original_image(self) -> None:
        set_ndarray_on_qlabel(self.img, self.ui.image_label)

    def display_processed_image(self, img: np.ndarray, format: QImage.Format = QImage.Format.Format_Grayscale8) -> None:
        self.processed = img
        self.processed_format = format
        set_ndarray_on_qlabel(img, self.ui.processed_label, format)

    def is_segmentation_done(self) -> bool:
        return self.seg is not None

    @Slot()
    def get_segmentation_result(self, seg: np.ndarray) -> None:
        self.seg = seg
        self.get_segmentation_signal.emit(self)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.display_original_image()
        if self.processed is not None and self.processed_format is not None:
            self.display_processed_image(self.processed, self.processed_format)
        return super().resizeEvent(event)
