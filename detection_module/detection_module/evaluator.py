import numpy as np
import cv2


class Evaluator:
    def __init__(self, iou_thresh=0.3):
        self.iou_thresh = iou_thresh
        self.tpfpfn = {}  # {'model1': {'tp': 0, 'fp': 0, 'fn': 0}, ...}

    def __call__(self, pred, grtr, model_name):
        splits = self.split_tpfpfn(grtr, pred)
        self.tpfpfn = self.accumulate(self.tpfpfn, self.update_counts(splits), model_name)
        recall, precision = self.get_recall_precision(model_name)
        return [recall, precision]

    def get_recall_precision(self, model_name):
        recall = self.tpfpfn[model_name]["grtr_tp"]/(self.tpfpfn[model_name]["grtr_tp"] + self.tpfpfn[model_name]["grtr_fn"])
        precision = self.tpfpfn[model_name]["pred_tp"]/(self.tpfpfn[model_name]["pred_tp"] + self.tpfpfn[model_name]["pred_fp"])
        return recall, precision

    def split_tpfpfn(self, grtr, pred):
        """
        :param pred: {'bbox': (M, 4), 'category': [N, 1], 'object': ...}
        :param grtr: same pred
        :return:
        """
        pred = self.insert_batch(pred)
        grtr = self.insert_batch(grtr)
        batch, M, _ = pred["category"].shape
        valid_mask = grtr["object"]
        ctgr_match = np.isclose(grtr["category"], np.swapaxes(pred["category"], 1, 2))  # (batch, N, M)
        ctgr_match = ctgr_match.astype(np.float32)
        iou = self.compute_iou_general(grtr["bboxes"], pred["bboxes"])  # (batch, N, M)
        iou *= ctgr_match
        best_iou = np.max(iou, axis=-1, keepdims=True)  # (batch, N, 1)
        best_idx = np.argmax(iou, axis=-1, keepdims=True)  # (batch, N, 1)
        iou_match = best_iou > self.iou_thresh  # (batch, N, 1)
        grtr_tp_mask = iou_match  # (batch, N, 1)
        grtr_fn_mask = ((1 - grtr_tp_mask) * valid_mask).astype(np.float32)  # (batch, N, 1)
        grtr_tp = {key: val * grtr_tp_mask for key, val in grtr.items()}
        grtr_fn = {key: val * grtr_fn_mask for key, val in grtr.items()}
        grtr_tp["iou"] = best_iou * grtr_tp_mask
        grtr_fn["iou"] = best_iou * grtr_fn_mask
        # last dimension rows where grtr_tp_mask == 0 are all-zero
        pred_tp_mask = self.indices_to_binary_mask(best_idx, grtr_tp_mask, M)
        pred_fp_mask = 1 - pred_tp_mask  # (batch, M, 1)
        pred_tp = {key: val * pred_tp_mask for key, val in pred.items()}
        pred_fp = {key: val * pred_fp_mask for key, val in pred.items()}
        return {"pred_tp": pred_tp, "pred_fp": pred_fp, "grtr_tp": grtr_tp, "grtr_fn": grtr_fn}

    def insert_batch(self, data):
        for key in data.keys():
            data[key] = np.expand_dims(data[key], axis=0)
        return data

    def compute_iou_general(self, grtr, pred):
        grtr = np.expand_dims(grtr, axis=-2)  # (batch, N1, 1, D1)
        pred = np.expand_dims(pred, axis=-3)  # (batch, 1, N2, D2)
        inter_tl = np.maximum(grtr[..., :2], pred[..., :2])  # (batch, N1, N2, 2)
        inter_br = np.minimum(grtr[..., 2:4], pred[..., 2:4])  # (batch, N1, N2, 2)
        inter_hw = inter_br - inter_tl  # (batch, N1, N2, 2)
        inter_hw = np.maximum(inter_hw, 0)
        inter_area = inter_hw[..., 0] * inter_hw[..., 1]  # (batch, N1, N2)

        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1]) # (batch, 1, N2)
        grtr_area = (grtr[..., 2] - pred[..., 0]) * (grtr[..., 3] - pred[..., 1])  # (batch, N1, 1)
        iou = inter_area / (pred_area + grtr_area - inter_area + 1e-5)  # (batch, N1, N2)
        return iou

    def create_categoriezed_mask(self, pred, grtr):
        max_length = max(len(pred), len(grtr))
        padded_pred = np.pad(pred, (0, max_length - len(pred)), mode="constant", constant_values=-1)
        padded_grtr = np.pad(grtr, (0, max_length - len(grtr)), mode="constant", constant_values=-1)
        categoriezed_mask = padded_pred == padded_grtr
        return categoriezed_mask

    def draw_bboxes(self, image, bboxes_list, delay=0):
        color_list = np.eye(3) * 255
        for color, bboxes in enumerate(bboxes_list):
            for bbox in bboxes:
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              tuple(color_list[color]))
        cv2.imshow("image", image)
        cv2.waitKey(delay)

    def indices_to_binary_mask(self, best_idx, valid_mask, depth):
        best_idx_onehot = self.one_hot(best_idx[..., 0], depth) * valid_mask
        binary_mask = np.expand_dims(np.max(best_idx_onehot, axis=1), axis=-1) # (batch, M, 1)
        return binary_mask.astype(np.float32)

    def count_per_class(self, boxes, mask, num_ctgr):
        boxes_ctgr = boxes["category"][..., 0].astype(np.int32)  # (batch, N')
        boxes_onehot = self.one_hot(boxes_ctgr, num_ctgr) * mask
        boxes_count = np.sum(boxes_onehot, axis=(0, 1))
        return boxes_count

    def one_hot(self, grtr_category, category_shape):
        one_hot_data = np.eye(category_shape)[grtr_category.astype(np.int32)]
        return one_hot_data

    def numpy_gather(self, params, index, dim=0):
        if dim is 1:
            batch_list = []
            for i in range(params.shape[0]):
                batch_param = params[i]
                batch_index = index[i]
                batch_gather = np.take(batch_param, batch_index)
                batch_list.append(batch_gather)
            gathar_param = np.stack(batch_list)
        else:
            gathar_param = np.take(params, index)
        return gathar_param

    def accumulate(self, existing, new, model_name):
        if model_name in existing.keys():
            for key, value in existing[model_name].items():
                if key in existing[model_name].keys():
                    existing[model_name][key] += value
        else:
            existing[model_name] = {k: 0 for k in new.keys()}
            for key, value in new.items():
                existing[model_name][key] = value
        return existing

    def update_counts(self, split):
        counts = {k: 0 for k in list(split.keys())}
        for atr in split.keys():
            if atr == "pred_tp" or atr == "pred_fp":
                counts[atr] = np.count_nonzero(split[atr]["scores"])
            elif atr == "grtr_tp" or atr == "grtr_fn":
                counts[atr] = np.sum(split[atr]["object"])
        return counts
