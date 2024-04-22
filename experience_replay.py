import numpy as np
import torch
from collections import deque
import random


class ExperiencePool(object):
    def __init__(self):
        self._fields = ["states", "actions", "returns"]
        self.initialize_memory()
        
    def initialize_memory(self):
        self.memory = {field: [] for field in self._fields}
    
    def insert_transitions(self, transitions):
        for field in self._fields:
            self.memory[field].extend(transitions[field])
        

    def sample_tensor(self, device: str) -> (
            tuple)[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, returns = self.memory["states"], self.memory["actions"], self.memory["returns"]

        return (torch.tensor(states, dtype=torch.float32, device=device),
                torch.tensor(np.array(actions), dtype=torch.float32, device=device),
                torch.tensor(returns, dtype=torch.float32, device=device))
        

    def clear(self):
        self.initialize_memory()
        


class NStepExperienceReplay:
    def __init__(self):
        self.initialize_memory()
    
    def initialize_memory(self):
        self.memory = {
            'state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'done': [0.0],
            'return': []
        }
        
    def push(self, state, action, reward, next_state, done):
        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['reward'].append(reward)
        self.memory['next_state'].append(next_state)
        self.memory['done'].append(done)
        self.memory['return'].append(None)
    
        
    def convert_to_n_step(self, next_value, gamma):
        self.memory['return'].append(next_value)
        for step in reversed(range(len(self.memory['reward']))):
            self.memory['return'][step] = self.memory['reward'][step] + gamma * self.memory['return'][step + 1] * (1 - self.memory['done'][step + 1])
            
        
        transitions = {
            'state': self.memory['state'],
            'action': self.memory['action'],
            'return': self.memory['return'][:-1]
        }
        return transitions
    
    def clear(self):
        self.initialize_memory()
        
