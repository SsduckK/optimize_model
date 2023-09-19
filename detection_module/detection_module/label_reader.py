import numpy as np
import os.path as op
import json


class LabelReader:
    def __init__(self, data_path):
        self.data_path = data_path

    def __call__(self, image_name):
        self.label_name = self.load_label(image_name)
        return self.label_name

    def load_label(self, file):
        label_name = file.replace("png", "json")
        label_file = op.join(self.data_path, label_name)
        bboxes = []
        categories = []
        objs = []
        with open(label_file, 'r') as f:
            label_data = json.load(f)
            for instance in label_data.keys():
                category, bbox, obj = self.extract_box(label_data[instance])
                if bbox is not None:
                    bboxes.append(bbox)
                    categories.append(category)
                    objs.append(obj)
        if label_data.__len__() == 0:
            bboxes = np.zeros((0, 4))
        else:
            bboxes = np.array(bboxes)
        categories = np.expand_dims(np.array(categories), axis=-1)
        objs = np.expand_dims(np.array(objs), axis=-1)
        return {"category": categories, "bboxes": bboxes, "object": objs}

    def extract_box(self, instance):
        category_name = instance[0]
        category_id = self.remapping_category2id(category_name)
        bbox = np.array(instance[1:-1], dtype=np.int32)
        obj = instance[-1]
        return category_id, bbox, obj

    def remapping_category2id(self, category):
        class_id = {"person": 0, "rider": 0, "car": 1,
                    "truck": 2, "bus": 3, "train": 4,
                    "motorcycle": 5, "bicycle": 6}
        id = class_id[category]
        return id
