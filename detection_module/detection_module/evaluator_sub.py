import pandas as pd
import numpy as np


class EvaluatorFrame:
    def __init__(self, model_num, compression_num):
        self.evaluator_frame_base = self.create_frame_base(model_num, compression_num)
        self.evaluator_frame = pd.DataFrame(self.evaluator_frame_base,
                                            columns=["model", "compression", "tp", "fp", "fn"])

    def create_frame_base(self, model_num, comp_num):
        frame_base = np.zeros((model_num * comp_num, 5), dtype=np.int16)
        row_idx = 0
        for model_num_idx in range(model_num):
            for comp_num_idx in range(comp_num):
                frame_base[row_idx][0] = model_num_idx
                frame_base[row_idx][1] = comp_num_idx
                row_idx += 1
        return frame_base

    def update_data(self, model_index, compression_index, tpfpfn):
        splits = [tpfpfn["tp"], tpfpfn["fp"], tpfpfn["fn"]]
        condition = ((self.evaluator_frame["model"] == model_index) &
                     (self.evaluator_frame["compression"] == compression_index))
        self.evaluator_frame.loc[condition, ["tp", "fp", "fn"]] += splits

    def get_row(self, model_index, compression_index):
        condition = ((self.evaluator_frame["model"] == model_index) &
                     (self.evaluator_frame["compression"] == compression_index))
        return self.evaluator_frame.loc[condition].to_numpy()[0]


def main():
    model = 3
    compression = 3
    evaluator_frame = EvaluatorFrame(model, compression)


if __name__ == "__main__":
    main()
