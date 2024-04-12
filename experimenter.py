import re
from typing import Iterable

import gymnasium as gym
import pettingzoo
import rware
# from lbforaging.foraging.pettingzoo_environment import parallel_env
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from agent import ACAgent, SEACAgent


class Experimenter(object):
    def __init__(self, env: gym.Env, agents: list[ACAgent], save_path: str, episode_max_length: int = 500):
        self.env = env
        self.env.reset() # required to instantiate some parts of pettingzoo envs
        self.agents = agents
        if isinstance(env, pettingzoo.utils.conversions.aec_to_parallel_wrapper):
            self.agent_dict = {env.aec_env.agents[i] : agents[i] for i in range(len(agents))}
        else:
            self.agent_dict = {i: agents[i] for i in range(len(agents))}
        self.episode_max_length = episode_max_length
        self.save_path = save_path
        self.experiment_history = {
            "episode": [],
            "mean_reward": [],
            "std_reward": [],
            "mean_length": [],
            "std_length": []
        }
        # TODO: set seed for random, torch, np etc.

    def generate_episode(self, render: bool = False, training: bool = True) -> tuple[np.float64, int]:
        states, info = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            actions = [agent.act(states[agent_id], training=training) for agent_id, agent in self.agent_dict.items()]
            # TODO: Add truncated and terminated return vals for old gym envs
            next_states, rewards, terminated, truncated, info = self.env.step(actions)

            done = np.all(terminated) or np.all(truncated) or episode_length > self.episode_max_length

            if training:
                for (agent_id, agent) in self.agent_dict.items():
                    agent.remember(states[agent_id], actions[agent_id], rewards[agent_id],
                                   next_states[agent_id], done and 1 or 0)

            states = next_states
            episode_length += 1
            episode_reward += np.sum(rewards)

            if render:
                self.env.render()
                
        # n-step TD learning
        if training:
            for agent in self.agents:
                agent.memory.convert_to_n_step(agent.n_steps, agent.gamma)

        return episode_reward, episode_length

    def learn(self, num_steps: int = 50):
        for agent_id, agent in enumerate(self.agents):
            agent.learn(num_steps=num_steps)

    def clear_experience(self):
        for agent in self.agents:
            agent.memory.clear()

    def evaluate_policies(self, num_repetitions: int = 10, render: bool = False) -> dict:
        for agent in self.agents:
            agent.eval()

        rewards, lengths = [], []
        for i in range(num_repetitions):
            episode_reward, episode_length = self.generate_episode(training=True, render=(i == 0 and render))
            rewards.append(episode_reward)
            lengths.append(episode_length)

        for agent in self.agents:
            agent.train()

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths)
        }

    def run(self, args):
        num_episodes = args.total_env_steps // self.episode_max_length
        for episode in tqdm(range(num_episodes)):
            if episode < args.warmup_episodes:
                self.generate_episode(training=True)
            else:
                reward, _ = self.generate_episode(training=True)
                # TODO: logger?
                if args.verbose > 0:
                    print(f"Episode {episode}: {reward}")
                self.learn(num_steps=args.num_gradient_steps)
                self.clear_experience()

                if episode % args.evaluate_frequency == 0 or episode == num_episodes - 1:
                    result = self.evaluate_policies(args.evaluate_episodes, render=args.render)
                    self.experiment_history["episode"].append(episode)
                    for key, value in result.items():
                        print(f"{key}: {value}", end=" ")
                        self.experiment_history[key].append(value)
                    print()
                    for agent_id, agent in enumerate(self.agents):
                        agent.save(f"{self.save_path}/agent_{agent_id}.pth")

                    # save results periodically
                    mean_rewards = np.array(self.experiment_history["mean_reward"])
                    std_rewards = np.array(self.experiment_history["std_reward"])
                    x_axis = np.array(self.experiment_history["episode"]) * args.evaluate_episodes
                    plt.plot(x_axis, mean_rewards)
                    plt.fill_between(x_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
                    plt.xlabel("Time Steps")
                    plt.ylabel("Episode Rewards")
                    plt.savefig(f"{self.save_path}/results.svg")
                    plt.close()
                    np.save(f"{self.save_path}/experiment_history.npy", self.experiment_history)


implemented_agent_types = ["IAC", "SEAC"]


def create_experiment(args) -> Experimenter:
    agent_type: str = args.agent_type
    env_name: str = args.env
    num_agents: int = args.num_agents
    episode_max_length: int = args.episode_max_length
    device: torch.device = args.device
    se_lambda_value: float = args.SEAC_lambda_value
    save_path: str = args.save_path
    batch_size: int = args.batch_size
    n_steps: int = args.n_steps

    assert (agent_type in implemented_agent_types)

    # Handle different env types:
    if "foraging" in env_name.lower():
        pattern = r'^Foraging-(\d+)x(\d+)-(\d+)p-(\d+)f-(\w+)$'
        # Match the pattern against the environment name
        match = re.match(pattern, env_name)
        assert match
        field_size = (int(match.group(1)), int(match.group(2)))  # Shape as a tuple
        players = int(match.group(3))  # Extract the number of players
        max_food = int(match.group(4))
        # Magic numbers, these are default params in SEAC paper.
        max_level = 3  # magic number, but this is what they use in paper.
        sight = field_size[0]
        max_episode_steps = episode_max_length  # Default is 50!
        force_coop = False
        env = parallel_env(players=players,
                           max_player_level=max_level,
                           field_size=field_size,
                           max_food=max_food,
                           sight=sight,
                           max_episode_steps=max_episode_steps,
                           force_coop=force_coop)
    else:
        env = gym.make(env_name)

    # TODO: Support for multiple lists / teams (e.g. team based where a subset of agents have access to each other)
    # TODO: This is implemented in the 'if agent_type == ...' branches.
    agent_list = []

    # TODO: idk if todo, but would be nice to wrap rware in pettingzoo env.
    if isinstance(env, pettingzoo.ParallelEnv):
        agent0 = env.possible_agents[0]
        obs_space = env.observation_spaces[agent0].shape[0]
        action_space = env.action_spaces[agent0].n
    elif isinstance(env.observation_space, Iterable):
        obs_space = env.observation_space[0].shape[0]
        action_space = env.action_space[0].n
    else:
        raise ValueError(f"Unsupported env type: {type(env)}")

    # Individual agents with no access to each other
    if agent_type == "IAC":
        for i in range(num_agents):
            agent = ACAgent(obs_space, action_space,
                            episode_max_length=episode_max_length, device=device, batch_size=batch_size, n_steps=n_steps)
            agent_list.append(agent)

    # Several references to the same agent (shared network)
    elif agent_type == "SNAC":
        # TODO: update agent.py
        agent = ACAgent(obs_space, action_space,
                        episode_max_length=episode_max_length * num_agents, device=device, batch_size=batch_size, n_steps=n_steps)
        for i in range(num_agents):
            agent_list.append(agent)

    # Individual agents with access to each other
    elif agent_type == "SEAC":
        for i in range(num_agents):
            agent = SEACAgent(obs_space, action_space,
                              episode_max_length=episode_max_length, device=device,
                              agent_list=agent_list, lambda_value=se_lambda_value, batch_size=batch_size, n_steps=n_steps)
            agent_list.append(agent)

    if args.pretrain_path is not None:
        for agent_id, agent in enumerate(agent_list):
            agent.load(f"{args.pretrain_path}/agent_{agent_id}.pth")

    return Experimenter(env, agent_list, save_path, episode_max_length=episode_max_length)
