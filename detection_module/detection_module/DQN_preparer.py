class DqnInput:
    def __init__(self):
        self.meta_info = {"model": [0, 0, 0], "compression": [0, 0, 0],
                          "C2S_time": [0, 0, 0], "detection_time": [0, 0, 0]}

    def update(self, model=0, compression=90, C2S_time=0, detection_time=0):
        input_list = [model, compression, C2S_time, detection_time]
        for idx, key in enumerate(self.meta_info.keys()):
            self.meta_info[key].append(input_list[idx])
        if len(self.meta_info["model"]) > 10:
            self.meta_info = dict(map(lambda x: (x[0], self.slice_recent(x[1], 10)), self.meta_info.items()))

    def slice_recent(self, arr, slice_frame):
        return arr[-slice_frame:]

    def get(self, recent_frame=3):
        recent_info = dict(map(lambda x: (x[0], self.slice_recent(x[1], recent_frame)), self.meta_info.items()))
        return recent_info

