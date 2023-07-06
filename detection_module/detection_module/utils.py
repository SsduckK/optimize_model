import numpy as np
import os.path as op


def load_label(file):
    label_name = file.replace("png", "txt")
    label_file = op.join("/mnt/intHDD/kitti/training/label_2", label_name)
    bboxes = []
    categories = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            bbox, category = extract_box(line)
            if bbox is not None:
                bboxes.append(bbox)
                categories.append(category)
    bboxes = np.array(bboxes)
    return {"categories": categories, "bboxes": bboxes}


def extract_box(line):
    raw_label = line.strip("\n").split(" ")
    category = raw_label[0]
    y1 = round(float(raw_label[5]))
    x1 = round(float(raw_label[4]))
    y2 = round(float(raw_label[7]))
    x2 = round(float(raw_label[6]))
    bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
    return bbox, category


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

    pred_ctgr_aligned = np.take(pred["labels"], best_idx)
    ctgr_aligned = create_categoriezed_mask(pred_ctgr_aligned, category)
    grtr_tp_mask = np.expand_dims(iou_match * ctgr_aligned, axis=-1)


def create_categoriezed_mask(pred, grtr):
    max_length = max(len(pred), len(grtr))
    padded_pred = np.pad(pred, (0, max_length - len(pred)), mode="constant", constant_values=-1)
    padded_grtr = np.pad(grtr, (0, max_length - len(grtr)), mode="constant", constant_values=-1)
    print(padded_pred)
    print(padded_grtr)
    categoriezed_mask = padded_pred == padded_grtr
    return categoriezed_mask