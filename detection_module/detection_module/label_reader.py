import numpy as np
import os.path as op


class LabelReader:
    def __init__(self, data_path):
        self.data_path = data_path

    def __call__(self, image_name):
        self.label_name = self.load_label(image_name)
        return self.label_name

    def load_label(self, file):
        label_name = file.replace("png", "txt")
        label_file = op.join(self.data_path, label_name)
        bboxes = []
        categories = []
        objs = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                bbox, category, obj = self.extract_box(line)
                if bbox is not None:
                    bboxes.append(bbox)
                    categories.append(category)
                    objs.append(obj)
        categories = np.expand_dims(np.array(categories), axis=-1)
        bboxes = np.array(bboxes)
        objs = np.expand_dims(np.array(objs), axis=-1)
        return {"category": categories, "bboxes": bboxes, "object": objs}

    def extract_box(self, line):
        raw_label = line.strip("\n").split(" ")
        category = raw_label[0]
        id = self.remapping_category2id(category)
        y1 = round(float(raw_label[5]))
        x1 = round(float(raw_label[4]))
        y2 = round(float(raw_label[7]))
        x2 = round(float(raw_label[6]))
        bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
        return bbox, id, 1

    def remapping_category2id(self, category):
        class_id = {"Car": 2, "Van": 2, "Truck": 8,
                    "Pedestrian": 0, "Person_sitting": 0, "Cyclist": 0,
                    "Tram": 8, "Misc": 81, "DontCare": 82}
        id = class_id[category]
        return id
