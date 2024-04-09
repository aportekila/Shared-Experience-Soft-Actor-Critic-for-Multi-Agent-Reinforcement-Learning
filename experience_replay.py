import numpy as np
import torch
from collections import deque
import random


class ExperienceReplay(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.number, reward: np.float64, next_state: np.ndarray, done: int):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple[np.ndarray, np.number, np.float64, np.ndarray, int]]:
        return zip(*random.sample(self.memory, batch_size))

    def sample_tensor(self, batch_size: int, device: str) -> (
            tuple)[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, reward, next_state, done = self.sample(batch_size)

        return (torch.tensor(np.array(state), device=device),
                torch.tensor(np.array(action), device=device),
                torch.tensor(np.array(reward), device=device),
                torch.tensor(np.array(next_state), device=device),
                torch.tensor(np.array(done), device=device)
                )

    def __len__(self) -> int:
        return len(self.memory)

    def __repr__(self) -> str:
        return f'ExperienceReplay(capacity={self.capacity}, len={len(self.memory)})'
