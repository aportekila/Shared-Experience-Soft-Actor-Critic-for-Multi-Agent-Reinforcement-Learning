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

        return (torch.tensor(np.array(state), device=device, dtype=torch.float32),
                torch.tensor(np.array(action), device=device, dtype=torch.float32),
                torch.tensor(np.array(reward), device=device, dtype=torch.float32),
                torch.tensor(np.array(next_state), device=device, dtype=torch.float32),
                torch.tensor(np.array(done), device=device, dtype=torch.float32)
                )
    def clear(self):
        self.memory.clear()
        
    def __len__(self) -> int:
        return len(self.memory)

    def __repr__(self) -> str:
        return f'ExperienceReplay(capacity={self.capacity}, len={len(self.memory)})'


class EpisodicExperienceReplay(ExperienceReplay):
    def __init__(self, episode_max_length):
        super(EpisodicExperienceReplay, self).__init__(episode_max_length)
        self.episode_max_length = episode_max_length
        
    def convert_to_n_step(self, n_steps: int, gamma: float):
        assert n_steps >= 1, "n_steps must be greater than or equal to 1"
        if n_steps == 1:
            return
        
        for i in range(len(self.memory)):
            state, action, reward, next_state, done = self.memory[i]
            if done:
                return
            n_step_reward = reward
            for j in range(1, n_steps):
                if i+j >= len(self.memory):
                    break
                _, _, r, next_state, d = self.memory[i+j]
                n_step_reward += gamma**j * r
                if d:
                    break
            self.memory[i] = (state, action, n_step_reward, next_state, d)
            
        
        
    def __repr__(self) -> str:
        return f'EpisodicExperienceReplay(episode_max_length={self.episode_max_length})'
