import gymnasium as gym
import numpy as np

from agent import ACAgent


class Experimenter(object):
    def __init__(self, env: gym.Env, agents: list[ACAgent], episode_max_length: int=500, ):
        self.env = env
        self.agents = agents
        self.episode_max_length = episode_max_length

    def generate_episode(self, render: bool=False, training: bool=True) -> tuple[np.float64, int]:
        states, info = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            actions = []
            for idx, agent in enumerate(self.agents):
                action = agent.act(states[idx], training=training)
                actions.append(action)

            # TODO: Add truncated and terminated return vals for old gym envs
            next_states, rewards, dones, info = self.env.step(actions)

            if training:
                for idx, agent in enumerate(self.agents):
                    agent.remember(states[idx], actions[idx], rewards[idx], next_states[idx], int(dones[idx]))

            done = np.all(dones) or episode_length > self.episode_max_length
            states = next_states
            episode_length += 1
            episode_reward += np.sum(rewards)

            if render:
                self.env.render()
        return episode_reward, episode_length

    def learn(self):
        for agent in self.agents:
            agent.learn()

    def clear_experience(self):
        for agent in self.agents:
            agent.memory.memory.clear()

    def evaluate_policies(self, num_repetitions: int=10) -> np.float64:
        total_reward = 0
        for i in range(num_repetitions):
            episode_reward, _ = self.generate_episode(training=False)
            total_reward += episode_reward
        return total_reward/num_repetitions