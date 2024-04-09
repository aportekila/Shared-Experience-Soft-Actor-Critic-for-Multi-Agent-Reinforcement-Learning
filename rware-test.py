import rware
import gymnasium as gym
from tqdm import tqdm

from agent import ACAgent
import torch
import numpy as np


TOTAL_ENV_STEPS = 50000000
MAX_EPISODE_LENGTH = 500

env = gym.make("rware-small-4ag-v1")

# observation space is a tuple containing spaces for each agent
agents = [ACAgent(env.observation_space[i].shape[0], env.action_space[i].n, 5000, "cuda") for i in range(env.n_agents)]

# training loop
for i in tqdm(range(int(TOTAL_ENV_STEPS/MAX_EPISODE_LENGTH))):
    states, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    while not done:
        actions = []
        for idx, agent in enumerate(agents):
            action = agent.act(states[idx], training=True)
            actions.append(action)

        next_states, rewards, dones, info = env.step(actions)

        done = np.any(dones) or episode_length > MAX_EPISODE_LENGTH

        for idx, agent in enumerate(agents):
            agent.remember(states[idx], actions[idx], rewards[idx], next_states[idx], int(done))

        states = next_states
        episode_length += 1
        episode_reward += np.sum(rewards)

        # env.render()

    for agent in agents:
        agent.learn(num_steps=10)
        agent.memory.memory.clear()
        if i % 1000 == 0:
            agent.save(f"weights_{i}.pth")
