import numpy as np
import pandas as pd
import os
import os.path as op
import datetime
import json

from . import config as cfg


class LoggingTool:
    def __init__(self, mean_range, save_path=None):
        self.frame_idx = 0
        self.base_frame = self.load_frame()
        self.validate_state = self.load_validate_state(cfg.VALIDATING_DATA)
        self.mean_range = mean_range
        self.save_path = save_path
        self.target_time = cfg.TARGET_TIME
        self.loss = 0
        self.time_list = []
        self.time_diff = 0
        self.F1_score = 0

    def load_frame(self):
        return np.array([["idx", "loss", "time_diff", "F1 score"]])

    def load_validate_state(self, val_file):
        with open(val_file, 'r') as f:
            val_data = json.load(f)
        return np.array(val_data["data"])

    def get_loss(self, loss):
        self.loss = loss

    def get_time_diff(self, time):
        if len(self.time_list) > self.mean_range:
            self.time_list = self.time_list[1:]
            mean_time = self.calculating_time(self.time_list)
            self.time_diff = cfg.TARGET_TIME - mean_time
        self.time_list.append(time)

    def calculating_time(self, time_list):
        time_list = np.array(time_list)
        mean_time = np.mean(time_list)
        return mean_time

    def get_F1score(self, F1_score):
        self.F1_score = F1_score

    def record_validation(self, actions):
        actions = np.asarray(actions).squeeze()
        episode_list = np.array([["|"] * actions.shape[0]]).transpose()
        self.validate_state = np.concatenate([self.validate_state, episode_list, actions], axis=1)

    def logging(self):
        loss = self.loss.cpu().detach().numpy()
        frame_info = np.array([[self.frame_idx, loss, self.time_diff, self.F1_score]])
        print(f"Loss : {self.loss}, Time diff : {self.time_diff}, F1 score : {self.F1_score}")
        self.base_frame = np.concatenate((self.base_frame, frame_info), axis=0)
        self.frame_idx += 1

    def saving_data(self, path):
        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        new_path = op.join(path, date_time_str)
        os.mkdir(new_path)
        loss_timediff_f1score = pd.DataFrame(self.base_frame)
        validate_result = pd.DataFrame(self.validate_state)
        loss_timediff_f1score.to_csv(op.join(new_path, "log.csv"), index=False)
        validate_result.to_csv(op.join(new_path, "validate.csv"), index=False)


def main():
    np.random.seed(1)
    test = "1"
    log = LoggingTool(test, 10)
    sample_time = [np.random.randint(30, 35) * 0.001 for _ in range(100)]
    sample_loss = [np.random.random() for _ in range(100)]
    for t, l in zip(sample_time, sample_loss):
        log.get_time_diff(t)
        log.get_loss(l)
        log.logging()
    df = pd.DataFrame(log.base_frame)
    df.to_csv("/home/gorilla/lee_ws/ros/src/optimize_model/detection_module/detection_module/data/sample_log.csv")


if __name__ == "__main__":
    main()
