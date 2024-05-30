import re
from typing import Iterable

import gymnasium as gym
import pettingzoo
import rware
import numpy as np
import torch
from tqdm import tqdm
import concurrent
import matplotlib.pyplot as plt

from environments import ProjectBaseEnv, RwareEnvironment, ForagingEnvironment, PettingZooEnvironment
from agent import ACAgent, SEACAgent, SNACAgent
from experience_replay import EpisodicExperienceReplay
from utils import seed_everything


class Experimenter(object):
    def __init__(self, args, env: ProjectBaseEnv, agents: list[ACAgent], save_path: str):
        self.args = args
        self.env = env
        self.learning_agents = agents
        self.agent_names = self.env.agents
        self.save_path = save_path
        self.n_agents = len(self.agent_names)
        self.experiment_history = {
            "episode": [],
            "mean_reward": [],
            "std_reward": [],
            "mean_length": [],
            "std_length": []
        }

        self.temp_memories = []  # for n-step TD learning

    def generate_episode(self, render: bool = False, training: bool = True) -> tuple[np.float64, int]:
        self.temp_memories.append(
            {agent_id: EpisodicExperienceReplay(self.env.max_steps) for agent_id in self.agent_names})
        states, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        # This is pettingzoo convention. List will be empty when done.
        while self.env.agents:
            # States is a dictionary with agent name as key
            actions = {agent_id: agent.act(states[agent_id], training=training)
                       for agent, agent_id in zip(self.learning_agents, self.agent_names)}

            next_states, rewards, terminated, truncated, info = self.env.step(list(actions.values()))

            done = np.all(list(terminated.values())) or np.all(list(truncated.values()))

            if training:
                for agent_id, memory in self.temp_memories[-1].items():
                    memory.push(states[agent_id], actions[agent_id], rewards[agent_id],
                                next_states[agent_id], done and 1 or 0)

            states = next_states
            episode_length += 1
            episode_reward += np.sum(list(rewards.values()))

            if render:
                self.env.render()

        # n-step TD learning
        if training:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_agents) as executor:
                for agent_id, memory in enumerate(self.temp_memories[-1].values()):
                    executor.submit(memory.convert_to_n_step, n_steps=self.learning_agents[agent_id].n_steps,
                                    gamma=self.learning_agents[agent_id].gamma)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_agents) as executor:
                for agent_id, memory in enumerate(self.temp_memories[-1].values()):
                    executor.map(self.learning_agents[agent_id].remember_tuple, memory.memory)

        return episode_reward, episode_length

    def learn(self, num_steps: int = 50):
        for agent_id, agent in enumerate(self.learning_agents):
            agent.learn(num_steps=num_steps)

    def clear_experience(self):
        self.temp_memories = []
        for agent in self.learning_agents:
            agent.memory.clear()

    def evaluate_policies(self, num_repetitions: int = 10, render: bool = False) -> dict:
        for agent in self.learning_agents:
            agent.eval()

        rewards, lengths = [], []
        for i in range(num_repetitions):
            episode_reward, episode_length = self.generate_episode(training=True, render=(i == 0 and render))
            rewards.append(episode_reward)
            lengths.append(episode_length)

        for agent in self.learning_agents:
            agent.train()

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths)
        }

    def run(self, args):
        num_episodes = args.total_env_steps // self.env.max_steps
        for episode in tqdm(range(num_episodes)):
            if episode < args.warmup_episodes:
                self.generate_episode(training=True)
            else:
                reward, length = self.generate_episode(training=True)
                if args.verbose > 0:
                    print(f"Episode {episode}: {reward}, length: {length}")

                if episode % args.update_frequency == 0 and len(self.learning_agents[0].memory) > self.args.batch_size:
                    self.learn(num_steps=args.num_gradient_steps)
                    self.clear_experience()

                if episode % args.evaluate_frequency == 0 or episode == num_episodes - 1:
                    result = self.evaluate_policies(args.evaluate_episodes, render=args.render)
                    self.experiment_history["episode"].append(episode)
                    for key, value in result.items():
                        print(f"{key}: {value}", end=" ")
                        self.experiment_history[key].append(value)
                    print()
                    for agent_id, agent in enumerate(self.learning_agents):
                        agent.save(f"{self.save_path}/agent_{agent_id}.pth")

                    # save results periodically
                    mean_rewards = np.array(self.experiment_history["mean_reward"])
                    std_rewards = np.array(self.experiment_history["std_reward"])
                    x_axis = np.array(self.experiment_history["episode"]) * self.env.max_steps
                    plt.plot(x_axis, mean_rewards)
                    plt.fill_between(x_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
                    plt.xlabel("Time Steps")
                    plt.ylabel("Episode Rewards")
                    plt.title(f"{args.env} - {args.agent_type}")
                    plt.savefig(f"{self.save_path}/results.png")
                    plt.savefig(f"{self.save_path}/results.svg")
                    plt.close()
                    np.save(f"{self.save_path}/experiment_history.npy", self.experiment_history)


implemented_agent_types = ["IAC", "SNAC", "SEAC", "SENAC"]


def create_experiment(args) -> Experimenter:
    agent_type: str = args.agent_type
    env_name: str = args.env
    device: torch.device = args.device
    se_lambda_value: float = args.SEAC_lambda_value
    save_path: str = args.save_path
    batch_size: int = args.batch_size
    capacity: int = args.buffer_size
    episode_max_length: int = args.episode_max_length
    n_steps: int = args.n_steps
    seed: int = args.seed

    assert (agent_type in implemented_agent_types)

    # Handle different env types:
    if "rware" in env_name.lower():
        env = RwareEnvironment(env_name=env_name, max_steps=episode_max_length)
    elif "foraging" in env_name.lower():
        env = ForagingEnvironment(env_name=env_name, max_steps=episode_max_length)
    elif "multiwalker" in env_name.lower():
        env = PettingZooEnvironment(env_name="multiwalker", max_steps=episode_max_length)
    elif "waterworld" in env_name.lower():
        env = PettingZooEnvironment(env_name="waterworld", max_steps=episode_max_length)
    else:
        raise NotImplementedError(f"Environment {env_name} not implemented")

    is_discrete = env.is_discrete
    # TODO: Support for multiple lists / teams (e.g. team based where a subset of agents have access to each other)
    agent_list = []

    env.reset(seed=seed)
    # Set seed for reproducibility
    seed_everything(seed)

    # Individual agents with no access to each other
    if agent_type == "IAC":
        for agent_id in env.agents:
            agent = ACAgent(env.observation_shapes[agent_id], env.action_shapes[agent_id],
                            capacity=capacity, device=device, batch_size=batch_size,
                            n_steps=n_steps, is_discrete=is_discrete)
            agent_list.append(agent)
    # Agents use a single actor network, and the master calculates loss from all memories
    elif agent_type == "SNAC":
        num_agents = len(env.agents)
        master = SNACAgent(env.observation_shapes[env.agents[0]], env.action_shapes[env.agents[0]],
                           capacity=capacity * num_agents, device=device, master=None, agent_list=agent_list,
                           batch_size=batch_size, n_steps=n_steps, is_discrete=is_discrete)
        agent_list.append(master)
        for i in range(1, num_agents):
            agent_id = env.agents[i]
            agent = SNACAgent(env.observation_shapes[agent_id], env.action_shapes[agent_id],
                              capacity=capacity, device=device, master=master, agent_list=agent_list,
                              batch_size=batch_size, n_steps=n_steps, is_discrete=is_discrete)
            agent_list.append(agent)
    # Several references to the same agent (shared network and experience)
    elif agent_type == "SENAC":
        num_agents = len(env.agents)
        agent = ACAgent(env.observation_shapes[env.agents[0]], env.action_shapes[env.agents[0]],
                        capacity=capacity * num_agents, device=device, batch_size=batch_size,
                        n_steps=n_steps, is_discrete=is_discrete)
        for i in range(num_agents):
            agent_list.append(agent)
    # Individual agents with access to each other
    elif agent_type == "SEAC":
        for agent_id in env.agents:
            agent = SEACAgent(env.observation_shapes[agent_id], env.action_shapes[agent_id],
                              capacity=capacity, device=device,
                              agent_list=agent_list, lambda_value=se_lambda_value, batch_size=batch_size,
                              n_steps=n_steps, is_discrete=is_discrete)
            agent_list.append(agent)

    if args.pretrain_path is not None:
        for agent_id, agent in enumerate(agent_list):
            agent.load(f"{args.pretrain_path}/agent_{agent_id}.pth")

    return Experimenter(args, env, agent_list, save_path)
