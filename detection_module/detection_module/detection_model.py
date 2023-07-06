# import argparse
# import os
# import os.path as osp
# import warnings
# from copy import deepcopy
#
# from mmengine import ConfigDict
# from mmengine.config import Config, DictAction
# from mmengine.runner import Runner
#
# from mmdet.engine.hooks.utils import trigger_visualization_hook
# from mmdet.evaluation import DumpDetResults
# from mmdet.registry import RUNNERS
# from mmdet.utils import setup_cache_size_limit_of_dynamo
#
#
# def main():
#     config_file = '/home/gorilla/lee_ws/mmdetector/custom/coco_cfg.py'
#     checkpoint_file = '/home/gorilla/lee_ws/mmdetector/mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
#
#     setup_cache_size_limit_of_dynamo()
#
#     # load config
#     cfg = Config.fromfile(config_file)
#     cfg.load_from = checkpoint_file
#     cfg.work_dir = "./result/"
#     cfg.tta_model = dict(
#         type='DetTTAModel',
#         tta_cfg=dict(
#             nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
#     if 'runner_type' not in cfg:
#         # build the default runner
#         runner = Runner.from_cfg(cfg)
#     else:
#         # build customized runner from the registry
#         # if 'runner_type' is set in the cfg
#         runner = RUNNERS.build(cfg)
#
#
#     runner.test()
#
#
# if __name__ == "__main__":
#     main()


#====

import cv2
import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector


# Specify the path to model config and checkpoint file
config_file = '/mnt/intHDD/mmdet_ckpt/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco.py'
checkpoint_file = '/mnt/intHDD/mmdet_ckpt/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# Test a single image and show the results
img = '/mnt/intHDD/BDD100K/bdd100k_images_100k/bdd100k/images/100k/val/b1c9c847-3bda4659.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
pred = result.pred_instances
bboxes = pred.bboxes
scores = pred.scores

bboxes = bboxes[scores > 0.3]
scores = scores[scores > 0.3]

# Show the results
img = mmcv.imread(img)
for box in bboxes:
    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
cv2.imshow("image", img)
cv2.waitKey()
img = mmcv.imconvert(img, 'bgr', 'rgb')

visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=True)

# Test a video and show the results
# Build test pipeline
model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

# visualizer has been created in line 31 and 34, if you run this demo in one notebook,
# you need not build the visualizer again.

# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# The interval of show (ms), 0 is block
wait_time = 1

video_reader = mmcv.VideoReader('video.mp4')

cv2.namedWindow('video', 0)

for frame in track_iter_progress(video_reader):
    result = inference_detector(model, frame, test_pipeline=test_pipeline)
    visualizer.add_datasample(
        name='video',
        image=frame,
        data_sample=result,
        draw_gt=False,
        show=False)
    frame = visualizer.get_image()
    mmcv.imshow(frame, 'video', wait_time)

cv2.destroyAllWindows()