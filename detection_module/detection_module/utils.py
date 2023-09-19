import numpy as np
import os.path as op
import cv2


def draw_bboxes(image, pred_bboxes, gt_bboxes, index, delay=0):
    drawn_image = image.copy()
    for bbox, category in zip(pred_bboxes["bboxes"], pred_bboxes["category"]):
        drawn_image = cv2.rectangle(drawn_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                    (255, 0, 0))
        drawn_image = cv2.putText(drawn_image, str(category[0]), (int(bbox[0]), int(bbox[1] + 10)), cv2.FONT_ITALIC, 1,
                                  (255, 0, 0))

    for bbox, category in zip(gt_bboxes["bboxes"], gt_bboxes["category"]):
        drawn_image = cv2.rectangle(drawn_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                    (0, 0, 255))
        drawn_image = cv2.putText(drawn_image, str(category[0]), (int(bbox[2]), int(bbox[3] + 10)), cv2.FONT_ITALIC, 1,
                                  (0, 0, 255))

    # cv2.imwrite("/home/gorilla/lee_ws/ros/src/optimize_model/detection_module/detection_module/data/image/" + str(index) + ".png",
    #             drawn_image)
