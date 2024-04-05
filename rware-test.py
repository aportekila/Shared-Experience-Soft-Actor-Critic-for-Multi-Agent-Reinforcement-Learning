import rware
import gymnasium as gym
from agent import ACAgent
import torch


env = gym.make("rware-small-4ag-v1")

# observation space is a tuple containing spaces for each agent
agents = [ACAgent(env.observation_space[i].shape[0], env.action_space[i].n, 5000, "cuda") for i in range(env.n_agents)]

