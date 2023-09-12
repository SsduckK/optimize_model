import numpy as np
import os.path as op
import cv2


def compute_iou_general(grtr, pred):
    pred_ex = np.expand_dims(pred, axis=-2)
    grtr_ex = np.expand_dims(grtr, axis=0)
    inter_tl = np.maximum(pred_ex[:, :, :2], grtr_ex[:, :, :2])
    inter_br = np.minimum(pred_ex[:, :, 2:4], grtr_ex[:, :, 2:4])
    inter_hw = inter_br - inter_tl
    inter_hw = np.maximum(inter_hw, 0)
    inter_area = inter_hw[:, :, 0] * inter_hw[:, :, 1]
    inter = np.concatenate([inter_tl, inter_br], axis=-1)

    pred_area = (pred_ex[:, :, 2] - pred_ex[:, :, 0]) * (pred_ex[:, :, 3] - pred_ex[:, :, 1])
    grtr_area = (grtr_ex[:, :, 2] - grtr_ex[:, :, 0]) * (grtr_ex[:, :, 3] - grtr_ex[:, :, 1])
    iou = inter_area / (pred_area + grtr_area - inter_area + 1e-6)
    iou_coord = inter[iou > 0.5]
    return iou, iou_coord


def get_confusionmatrix(iou, category, pred):
    best_iou = np.max(iou, axis=-1)
    iou_match = best_iou > 0.5
    best_idx = np.argmax(iou, axis=-1)
    print(iou)
    print("pred_label :", pred["labels"])
    print("best_iou :", best_iou)
    print("iou_match :", iou_match)
    print("best_idx :", best_idx)

    pred_ctgr_aligned = np.take(pred["labels"], best_idx)
    print("gt_category :", category, len(category))
    print("prediction_category :", pred["labels"], len(pred["labels"]))
    print("aligned_category :", pred_ctgr_aligned, len(pred_ctgr_aligned))
    ctgr_aligned = create_categoriezed_mask(pred_ctgr_aligned, category)
    print(ctgr_aligned)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_aligned, axis=-1)
    print(grtr_tp_mask)


def create_categoriezed_mask(pred, grtr):
    max_length = max(len(pred), len(grtr))
    padded_pred = np.pad(pred, (0, max_length - len(pred)), mode="constant", constant_values=-1)
    padded_grtr = np.pad(grtr, (0, max_length - len(grtr)), mode="constant", constant_values=-1)
    categoriezed_mask = padded_pred == padded_grtr
    return categoriezed_mask


def draw_bboxes(image, bboxes_list, delay=0):
    color_list = np.eye(3) * 255
    bboxes_list = [bboxes_list for _ in range(3)]
    for color, bboxes in enumerate(bboxes_list):
        for bbox in bboxes:
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          tuple(color_list[color]))
    cv2.imshow("image", image)
    cv2.waitKey(delay)
