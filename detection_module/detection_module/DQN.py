import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from . import config as cfg


class DQNNetwork(nn.Module):
    def __init__(self, n_state, n_actions):
        super(DQNNetwork, self).__init__()
        self.layer1 = nn.Linear(n_state, 128)
        self.layer2 = nn.Linear(128, 128)

        self.select_models = nn.Linear(128, n_actions)
        self.select_comp = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        select_model = F.relu(self.select_models(x))
        select_comp = F.relu(self.select_comp(x))
        return select_model, select_comp


class DQN:
    def __init__(self, memory):
        self.steps = 0
        if torch.cuda.is_available():
            self.num_episodes = 600
        else:
            self.num_episodes = 50
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.model_name = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(4, cfg.MODEL_NUM).to(
            self.device)  # [state, action, next_state, reward] , action : MODEL_NUM_4
        self.target_net = DQNNetwork(4, cfg.MODEL_NUM).to(
            self.device)  # [state, action, next_state, reward] , action : MODEL_NUM_4
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

    def split_input(self, input_data):
        state = torch.tensor([[input_data[i] for i in input_data.keys() if i != "reward"]],
                             dtype=torch.float32, device=self.device)
        reward = [input_data["reward"]]
        return state, reward

    def optimize_model(self, memory):
        if len(memory) < self.BATCH_SIZE:
            return
        sample_memory = memory.sample(self.BATCH_SIZE)
        # curr_state, next_state, reward = batch()
        # model_selection, compression = func(curr_state)
        cur_state, cur_compression, cur_reward = self.batched_memory(sample_memory)
        model_selection_action_batch = torch.unsqueeze(cur_state[:, 0], 1).type(torch.int64)
        next_state_index = self.get_next_state_index(sample_memory)
        next_sample_memory = self.get_next_state(memory.meta_info, next_state_index)
        next_state, next_compression, next_reward = self.batched_memory(next_sample_memory)
        non_final_mask = torch.tensor([~torch.all(torch.isnan(s)) for s in next_state], device='cuda', dtype=torch.bool)
        with torch.no_grad():
            next_model_action, next_compression_actions = self.target_net(next_state)
            next_actions_value = torch.cat([next_model_action.max(1)[0].unsqueeze(1),
                                            next_compression_actions.max(1)[0].unsqueeze(1)], dim=1)
            next_state_values = next_actions_value * non_final_mask.unsqueeze(1)
        expected_state_action_values = (next_state_values * self.GAMMA) + cur_reward
        # memory.meta_info.to_csv("/home/gorilla/lee_ws/ros/src/optimize_model/detection_module/detection_module/data/total_memory.csv", sep=",")
        # sample_memory.to_csv("/home/gorilla/lee_ws/ros/src/optimize_model/detection_module/detection_module/data/sample_memory.csv", sep=",")
        # next_sample_memory.to_csv("/home/gorilla/lee_ws/ros/src/optimize_model/detection_module/detection_module/data/next_state.csv", sep=",")
        sample_model, sample_compression = self.policy_net(cur_state)
        model_selection_values = sample_model.gather(1, model_selection_action_batch)
        compression_selection_values = sample_compression.gather(1, cur_compression)
        criterion = nn.SmoothL1Loss()
        model_loss = criterion(model_selection_values, expected_state_action_values[:, 0].unsqueeze(1))
        comp_loss = criterion(compression_selection_values, expected_state_action_values[:, 1].unsqueeze(1))
        total_loss = model_loss + comp_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        print("optimize_done")
        print(total_loss)

    def update_model(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                    1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def select_action(self, state):
        state = np.asarray([state[:4]], dtype=np.float32)
        state = torch.tensor(state, device=self.device)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1 * self.steps / self.EPS_DECAY)
        self.steps += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state)[0].max(1)[1].view(1, 1), self.policy_net(state)[1].max(1)[1].view(1, 1)
        else:
            return torch.randint(0, cfg.MODEL_NUM, (1, 1), device=self.device, dtype=torch.long), \
                torch.randint(0, 3, (1, 1), device=self.device, dtype=torch.long)

    def batched_memory(self, memory):
        columns = memory.columns.values
        batch = {col: torch.tensor([np.asarray(memory.loc[:, col].values, dtype=np.float32)], device=self.device)
                 for col in columns}
        cur_state_batch = torch.cat((batch["model"], batch["compression"], batch["C2S_time"],
                                     batch["det_time"]))
        cur_state_batch = torch.transpose(cur_state_batch, 0, 1)
        cur_comp_batch = torch.transpose(batch["compression"], 0, 1).type(torch.int64)
        cur_reward_batch = torch.transpose(batch["reward"], 0, 1)
        return cur_state_batch, cur_comp_batch, cur_reward_batch

    def get_next_state_index(self, transitions):
        next_state_index = np.array(transitions.index.values) + 1
        return next_state_index

    def get_next_state(self, memory, next_state_index):
        next_state = memory.iloc[next_state_index]
        return next_state
