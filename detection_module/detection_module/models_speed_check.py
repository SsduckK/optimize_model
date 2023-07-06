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


class SpeedLog:
    def __init__(self, ckpt_list, output, using_video=False):
        self.ckpt_list = ckpt_list
        if using_video:
            self.input_video = mmcv.VideoReader("/home/gorilla/lee_ws/mmdetector/mmdetection/demo/demo.mp4")
            self.frames = list(track_iter_progress(self.input_video))
        else:
            self.frames = self.load_frames()
        self.items = None
        self.test_pipeline = None
        self.models = self.build_model()
        self.logging_data(output=output, vis=False)

    def build_model(self):
        self.items = {glob.glob(op.join(config, "*.py"))[0]: glob.glob(op.join(config, "*.pth"))[0] for config in self.ckpt_list}
        models = [init_detector(config, pth, device='cuda:0') for config, pth in self.items.items()]
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
        result = inference_detector(model, frame)
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


ckpt_list = [t for t in glob.glob(op.join("/mnt/intHDD/mmdet_ckpt/yolov7", "*")) if not "_base_" in t]
output_path = "/home/gorilla/lee_ws/optimize_model/optimize_model/speed_log"
t = SpeedLog(ckpt_list, output_path, using_video=True)
