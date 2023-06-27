import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class Predictor():
    def __init__(
        self,
        model_path='./nnUNet_results/Dataset001_SJTUThyroidNodule/nnUNetTrainer__nnUNetPlans__2d',
        checkpoint_name='checkpoint_best.pth'
    ):
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=False,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        # initializes the network architecture, loads the checkpoint
        self.predictor.initialize_from_trained_model_folder(
            model_path,
            use_folds=('all',),
            checkpoint_name=checkpoint_name,
        )


    def get_segmentation(self, img: np.ndarray) -> np.ndarray:
        return self.predictor.predict_single_npy_array(img, {'spacing': (999, 1, 1)}, None, None, False) # type: ignore
