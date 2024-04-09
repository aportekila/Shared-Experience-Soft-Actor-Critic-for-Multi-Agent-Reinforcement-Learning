import rware
import gymnasium as gym
from agent import ACAgent
import torch
import numpy as np

MAX_EPISODE_LENGTH = 100

env = gym.make("rware-small-4ag-v1")

# observation space is a tuple containing spaces for each agent
agents = [ACAgent(env.observation_space[i].shape[0], env.action_space[i].n, 5000, "cuda") for i in range(env.n_agents)]

# training loop
for i in range(100):
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

        env.render()

    print(episode_length, episode_reward)
    for agent in agents:
        agent.learn()
        agent.memory.memory.clear()