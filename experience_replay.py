import torch
from collections import deque
import random

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def sample_tensor(self, batch_size, device):
        state, action, reward, next_state, done = self.sample(batch_size)
        return torch.tensor(state, device=device), torch.tensor(action, device=device), torch.tensor(reward, device=device), torch.tensor(next_state, device=device), torch.tensor(done, device=device)
    
    def __len__(self):
        return len(self.memory)
    
    def __repr__(self):
        return f'ExperienceReplay(capacity={self.capacity}, len={len(self.memory)})'
    