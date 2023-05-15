#!/home/gorilla/.pyenv/versions/mmdetection/bin python3

import os
import os.path as op
import glob
import cv2
import time
import numpy as np
import csv

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector


class MainDetector:
    def __init__(self, ckpt_list):
        self.models = self.build_model(ckpt_list)
        self.test_pipeline = None

    def __call__(self, image):
        chosen_idx = self.optimize_model(self.models)
        detected_result = inference_detector(self.models[chosen_idx], image, self.test_pipeline[chosen_idx])
        return detected_result

    def build_model(self, ckpt_list):
        items = {glob.glob(op.join(config, "*.py"))[0]: glob.glob(op.join(config, "*.pth"))[0] for config in ckpt_list}
        models = [init_detector(config, pth) for config, pth in items.items()]
        for i, model in enumerate(models):
            model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
            self.test_pipeline[i] = Compose(model.cfg.test_dataloader.dataset.pipeline)
        return models

    def optimize_model(self, models):   #select best model and return models' index ==>NEED implement
        return 0