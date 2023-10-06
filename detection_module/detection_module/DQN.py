import math
import random
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from . import config as cfg
import pdb


class DQNNetwork(nn.Module):
    def __init__(self, n_state, n_actions):
        super(DQNNetwork, self).__init__()
        self.layer1 = nn.Linear(n_state, 128)
        self.layer2 = nn.Linear(128, 128)

        self.select_models = nn.Linear(128, n_actions)
        self.select_comp = nn.Linear(128, 3)

    def forward(self, x):
        #print("input", x)
        x = F.leaky_relu(self.layer1(x))
        #print("1st", x)
        x = F.leaky_relu(self.layer2(x))
        #print("2nd", x)
        select_model = F.leaky_relu(self.select_models(x))
        select_comp = F.leaky_relu(self.select_comp(x))
        #print("model_", select_model)
        #print("comp_", select_comp)
        return select_model, select_comp


class DQN:
    def __init__(self, memory):
        torch.autograd.set_detect_anomaly(True)
        self.steps = 0
        if torch.cuda.is_available():
            self.num_episodes = 600
        else:
            self.num_episodes = 50
        self.BATCH_SIZE = cfg.BATCH_SIZE
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
        non_final_mask = torch.isnan(next_state).any(dim=1)
        non_final_next_states = next_state[~non_final_mask]
        next_state_values = torch.zeros((self.BATCH_SIZE, 2), device=self.device)

        with torch.no_grad():
            next_model_action, next_compression_actions = self.target_net(non_final_next_states)
            next_actions_value = torch.cat([next_model_action.max(1)[0].unsqueeze(1),
                                            next_compression_actions.max(1)[0].unsqueeze(1)], dim=1)
            next_state_values[~non_final_mask] = next_actions_value
        expected_state_action_values = (next_state_values * self.GAMMA) + cur_reward
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
        return total_loss
        # with open("/home/gorilla/lee_ws/ros/src/optimize_model/detection_module/detection_module/data/loss.txt", 'a') as f:
        #     f.write(f"\n+{str(total_loss)}")

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
            return torch.randint(0, cfg.MODEL_NUM, (1, 1), device=self.device, dtype=torch.float32), \
                torch.randint(0, 3, (1, 1), device=self.device, dtype=torch.float32)

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

    def validating(self):
        result = []
        parameter = self.load_validating_data(cfg.VALIDATING_DATA)
        for state in parameter["data"]:
            model_norm = (state[0] - cfg.PARAMETER["model"]["mean"]) / cfg.PARAMETER["model"]["std"]
            comp_norm = (state[0] - cfg.PARAMETER["compression"]["mean"]) / cfg.PARAMETER["compression"]["std"]
            c2s_norm = (state[0] - cfg.PARAMETER["c2s_time"]["mean"]) / cfg.PARAMETER["c2s_time"]["std"]
            det_norm = (state[0] - cfg.PARAMETER["det_time"]["mean"]) / cfg.PARAMETER["det_time"]["std"]
            st = torch.tensor([model_norm, comp_norm, c2s_norm, det_norm], device=self.device)
            result.append([self.policy_net(st)[0].max(0)[1].view(1, 1).cpu().numpy(),
                           self.policy_net(st)[1].max(0)[1].view(1, 1).cpu().numpy()])
        return result

    def load_validating_data(self, data):
        with open(data, 'r') as f:
            parameter = json.load(f)
        return parameter
