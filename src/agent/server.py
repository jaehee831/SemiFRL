import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import json # Retained for potential future use, though not directly used by the agent now

# --- 설정 (Hyperparameters) ---
# DQN 기본 설정
LR = 0.001        # 학습률
GAMMA = 0.99      # 할인율
EPSILON_START = 1.0 # 탐험 시작 값
EPSILON_END = 0.01  # 탐험 종료 값
EPSILON_DECAY = 200 # 탐험 감소율
MEMORY_CAPACITY = 10000 # 리플레이 버퍼 크기
BATCH_SIZE = 64     # 학습 배치 사이즈
TARGET_UPDATE = 10  # 타겟 네트워크 업데이트 주기

# 클라이언트 액션 스페이스 정의 - This global list is no longer directly used by the class
# but can serve as a default or example for instantiation.
# CLIENT_PSEUDO_THRESHOLD_ACTIONS = [0.7, 0.8, 0.9, 0.95, 0.98]
# CLIENT_EPOCH_ACTIONS =  # REMOVED

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- DQN 모델 (Q-Network) ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# --- 클라이언트 에이전트 DQN 구현 ---
class ServerAgentDQN:
    def __init__(self, state_dim, num_clients, device='cuda'):
        self.state_dim = state_dim
        self.device = device
        self.TARGET_UPDATE = TARGET_UPDATE

        self.action_space_size = num_clients
        self.policy_net = QNetwork(self.state_dim, self.action_space_size).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_space_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.steps_done = 0

    def _get_action_from_index(self, action_idx):
        # 단일 인덱스를 threshold 값으로 변환
        # Assumes action_idx is a valid index for self.threshold_actions
        return self.threshold_actions[action_idx]

    def _get_index_from_action(self, threshold_val):
        # threshold 값을 단일 인덱스로 변환
        # Assumes threshold_val is present in self.threshold_actions
        try:
            return self.threshold_actions.index(threshold_val)
        except ValueError:
            # Handle cases where the threshold_val might not be in the list,
            # though typically this method would be called with a known valid action.
            # For robustness, error handling or assumptions should be clear.
            raise ValueError(f"Threshold value {threshold_val} not found in agent's action list.")


    def choose_action(self, state):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1).indices.item()
        else:
            action_idx = random.randrange(self.action_space_size)

        # 선택된 인덱스로부터 threshold 값을 얻어 반환
        threshold_val = self._get_action_from_index(action_idx)
        return action_idx, threshold_val # Returns index and the actual threshold value

    def store_transition(self, state, action_idx, reward, next_state, done):
        self.memory.push(state, action_idx, reward, next_state, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        # (state, action_idx, reward, next_state, done) 튜플을 분리
        batch_state, batch_action_indices, batch_reward, batch_next_state, batch_done = zip(*transitions)

        state_batch = torch.FloatTensor(np.array(batch_state)).to(self.device)
        # batch_action_indices contains the indices of actions taken
        action_batch = torch.LongTensor(batch_action_indices).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch_next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch_done).unsqueeze(1).to(self.device)

        # 현재 상태의 Q-값
        #.gather(1, action_batch) selects the Q-value for the action taken
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # 다음 상태의 최대 Q-값 (타겟 네트워크 사용)
        next_q_values = self.target_net(next_state_batch).max(1).values.unsqueeze(1)
        
        # 다음 상태가 최종 상태가 아닐 경우에만 Q-값을 사용
        expected_q_values = reward_batch + GAMMA * next_q_values * (~done_batch)

        # 손실 계산 및 최적화
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None: # Ensure gradient exists before clamping
                param.grad.data.clamp_(-1, 1) # Gradient Clipping
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())