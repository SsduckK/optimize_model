import os.path as op
import glob
import time
import numpy as np

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector


class Detectors:
    def __init__(self, ckpt_list):
        self.models = self.load_detector(ckpt_list)

    def load_detector(self, ckpt_list):
        items = {glob.glob(op.join(config, "*.py"))[0]: glob.glob(op.join(config, "*.pth"))[0] for config in ckpt_list}
        models = [init_detector(config, pth) for config, pth in items.items()]
        return models

    def run(self, detector, image):
        start_time = time.time()
        result = inference_detector(detector, image)
        detection_time = time.time() - start_time
        pred = result.pred_instances
        scores = pred.scores
        bboxes = pred.bboxes
        labels = pred.labels
        bboxes = bboxes[scores > 0.3].cpu().numpy()
        labels = np.expand_dims(labels[scores > 0.3].cpu().numpy(), axis=-1)
        scores = np.expand_dims(scores[scores > 0.3].cpu().numpy(), axis=-1)
        return {"bboxes": bboxes, "category": labels, "scores": scores}, detection_time
