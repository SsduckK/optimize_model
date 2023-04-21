import os
import os.path as op
import glob
import cv2
import time
import json
import csv

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import configs_list as cfglist

#
def load_models(cfg_ckpt_dict, need_models):
    models = {}
    count = 0
    for model in need_models:
        try:
            if len(cfg_ckpt_dict[model]) == 2:
                if '.py' in cfg_ckpt_dict[model][0]:
                    cfg = cfg_ckpt_dict[model][0]
                    CKPT = cfg_ckpt_dict[model][1]
                else:
                    cfg = cfg_ckpt_dict[model][1]
                    CKPT = cfg_ckpt_dict[model][0]
                models[model] = init_detector(cfg, CKPT, device='cuda:0')
                count += 1
        except:
            continue
    print("================num_model============= : ", count)
    return models

def load_cfg_ckpt_dict(path):
    cfg_ckpt_dict = {}
    a = os.listdir(path)
    for cfg in a:
        cfg_ckpt_dict[cfg] = glob.glob(f'{path}/{cfg}/*')
    return cfg_ckpt_dict

def load_models(cfg_ckpt_dict, need_models):
    models = {}
    count = 0
    for model in need_models:
        try:
            if len(cfg_ckpt_dict[model]) == 2:
                if '.py' in cfg_ckpt_dict[model][0]:
                    cfg = cfg_ckpt_dict[model][0]
                    CKPT = cfg_ckpt_dict[model][1]
                else:
                    cfg = cfg_ckpt_dict[model][1]
                    CKPT = cfg_ckpt_dict[model][0]
                models[model] = init_detector(cfg, CKPT, device='cuda:0')
                count += 1
        except:
            continue
    print("================num_model============= : ", count)
    return models

ckpt = glob.glob(op.join("/mnt/intHDD/mmdet_ckpt/test_yolo", "*"))

items = {glob.glob(op.join(config, "*.py"))[0]: glob.glob(op.join(config, "*.pth"))[0] for config in ckpt}

models = [init_detector(config, pth) for config, pth in items.items()]
test_pipeline = [i for i in range(6)]
for i, model in enumerate(models):
    model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
    test_pipeline[i] = Compose(model.cfg.test_dataloader.dataset.pipeline)


video_reader = mmcv.VideoReader('mmdetection/demo/demo.mp4')
frame_number = [i-1 for i in range(68)]
log_time = [[model_name.split('/')[-1].split('.')[0]] for model_name in items.keys()]
cv2.namedWindow('video', 0)
for idx in range(len(models)):
    for i, frame in enumerate(track_iter_progress(video_reader)):
        start = time.time()
        result = inference_detector(models[idx], frame, test_pipeline=test_pipeline[idx])
        inf_time = time.time() - start
        log_time[idx].append(inf_time)
    # visualizer.add_datasample(
    #     name='video',
    #     image=frame,
    #     data_sample=result,
    #     draw_gt=False,
    #     show=False)
#     frame = visualizer.get_image()
#     mmcv.imshow(frame, 'video', wait_time=1)
#
# cv2.destroyAllWindows()

print(log_time)
# with open(op.join(f"/home/gorilla/lee_ws/optimize_model/speed_log", time.strftime('%Y-%m-%d', time.localtime(time.time()))+"_faster-rcnn.json"), 'w') as f:
#     json.dump(log_time, f, indent=4)

with open(op.join(f"/home/gorilla/lee_ws/optimize_model/speed_log", time.strftime('%Y-%m-%d', time.localtime(time.time()))+"_frame-per-model.csv"), 'w') as f:
    w = csv.writer(f)
    w.writerow(frame_number)
    for log in log_time:
        w.writerow(log)
