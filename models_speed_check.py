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
from mmdetection.mmdet.registry import VISUALIZERS
from mmdetection.mmdet.apis import init_detector, inference_detector


class SpeedLog:
    def __init__(self, ckpt_list, output, using_video=False):
        self.ckpt_list = ckpt_list
        if using_video:
            self.input_video = mmcv.VideoReader("../mmdetection/demo/demo.mp4")
            self.frames = list(track_iter_progress(self.input_video))
        else:
            self.frames = self.load_frames()
        self.items = None
        self.test_pipeline = None
        self.models = self.build_model()
        self.logging_data(output=output, vis=False)

    def build_model(self):
        self.items = {glob.glob(op.join(config, "*.py"))[0]: glob.glob(op.join(config, "*.pth"))[0] for config in self.ckpt_list}
        models = [init_detector(config, pth) for config, pth in self.items.items()]
        self.test_pipeline = [i for i in range(len(models))]
        for i, model in enumerate(models):
            model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
            self.test_pipeline[i] = Compose(model.cfg.test_dataloader.dataset.pipeline)
        return models

    def load_frames(self):  #input = image directory, return image list/ NEED implement
        images = None
        return images

    def inf_frame_per_model(self):
        frame_per_model = [[model_name.split('/')[-1].split('.')[0]] for model_name in self.items.keys()]
        for idx in range(len(self.models)):
            for frame in self.frames:
                inf_time = self.check_speed(self.models[idx], frame, idx)
                frame_per_model[idx].append(inf_time)
        return frame_per_model

    def inf_model_per_frame(self):
        model_per_frame = [[model_name.split('/')[-1].split('.')[0]] for model_name in self.items.keys()]
        for frame in self.frames:
            for idx in range(len(self.models)):
                inf_time = self.check_speed(self.models[idx], frame, idx)
                model_per_frame[idx].append(inf_time)
        return model_per_frame

    def check_speed(self, model, frame, idx):
        start = time.time()
        inference_detector(model, frame, test_pipeline=self.test_pipeline[idx])
        inf_time = time.time() - start
        return inf_time

    def logging_data(self, output, vis=False):
        index = np.array([i - 1 for i in range(len(self.frames) + 1)])
        index = index[np.newaxis, :]
        split_bar = np.array(['|' for i in range(len(self.frames) + 1)])
        model_per_frame = np.array(self.inf_model_per_frame())
        frame_per_model = np.array(self.inf_frame_per_model())
        result_table = np.vstack([index, model_per_frame, split_bar[np.newaxis, :], frame_per_model])
        result_table = result_table.T
        self.write_log(output, result_table)

    def write_log(self, output, results):
        with open(op.join(output, time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))+".csv"), 'w') as f:
            w = csv.writer(f)
            for result in results:
                w.writerow(result)


ckpt_list = glob.glob(op.join("/mnt/intHDD/mmdet_ckpt/test_yolo", "*"))
output_path = "/home/gorilla/lee_ws/optimize_model/optimize_model/speed_log"
t = SpeedLog(ckpt_list, output_path, using_video=True)
print("")