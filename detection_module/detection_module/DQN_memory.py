import pandas as pd
import numpy as np
import random


class ReplayMemory:
    def __init__(self):
        self.meta_info = pd.DataFrame(
            {
                "model": [None] * 10000,
                "compression": [None] * 10000,
                "C2S_time": [None] * 10000,
                "det_time": [None] * 10000,
                "reward": [None] * 10000
            }
        )

    def __len__(self):
        non_none_rows = self.meta_info.dropna(subset=["model"])
        return len(non_none_rows)

    def append_data(self, frame, input_data):
        for key in input_data.keys():
            self.meta_info.at[frame, key] = input_data[key]

    def latest_state(self):
        last_valid_index = self.meta_info["model"].last_valid_index()

        if last_valid_index is not None:
            latest_state = self.meta_info.loc[last_valid_index]
            return latest_state.values
        else:
            return None

    def sample(self, size):
        non_none_indices = self.meta_info[self.meta_info["model"].notna()].index
        sample_index = random.sample(non_none_indices.tolist(), size)
        sample_list = self.meta_info.loc[sample_index]
        return sample_list


def main():
    memory = ReplayMemory()
    for i in range(5):
        input_data = {
            "model": np.array(np.random.randint(0, 5)),
            "compression": np.random.random(),
            "C2S_time": np.random.randint(0, 4),
            "det_time": np.random.randint(0, 2)
        }
        memory.append_data(input_data, i)
        next_state = {"reward" : np.array([np.random.randint(0, 5)])}
        memory.append_data(next_state, i - 1)
    sample_list = memory.sample(5)
    print(memory.latest_state())
    print(memory.latest_state()[0])


if __name__ == "__main__":
    main()