import pandas as pd
import numpy as np
import random


class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.meta_info = pd.DataFrame(
            {
                "model": [None] * max_size,
                "compression": [None] * max_size,
                "C2S_time": [None] * max_size,
                "det_time": [None] * max_size,
                "reward": [None] * max_size
            },
        )

    def __len__(self):
        non_none_rows = self.meta_info.dropna(subset=["model"])
        return len(non_none_rows)

    def append_data(self, frame, input_data):
        for key in input_data.keys():
            self.meta_info.at[frame, key] = input_data[key]
        if len(self.meta_info) > self.max_size:
            self.meta_info = self.meta_info.iloc[1:]
            self.meta_info = self.meta_info.set_index(pd.RangeIndex(start=0, stop=self.max_size))

    def latest_state(self):
        last_valid_index = self.meta_info["model"].last_valid_index()

        if last_valid_index is not None:
            latest_state = self.meta_info.loc[last_valid_index]
            return latest_state.values
        else:
            return None

    def sample(self, size):
        # non_none_indices = self.meta_info[self.meta_info["model"].notna()].index
        non_none_indices = self.meta_info.iloc[:-1][~self.meta_info.iloc[:-1]["model"].isna()].index
        sample_index = random.sample(non_none_indices.tolist(), size)
        sample_list = self.meta_info.loc[sample_index]
        return sample_list


def main():
    memory = ReplayMemory(3)
    for idx, i in enumerate(range(10)):
        input_data = {
            "model": np.array(np.random.randint(0, 5)),
            "compression": np.random.random(),
            "C2S_time": np.random.randint(0, 4),
            "det_time": np.random.randint(0, 2),
            "reward": np.random.random()
        }
        memory.append_data(idx, input_data)
        print(memory.meta_info)


if __name__ == "__main__":
    main()