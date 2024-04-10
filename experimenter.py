from typing import Literal

import gymnasium as gym
import rware
import numpy as np
from tqdm import tqdm

from agent import ACAgent, SEACAgent




class Experimenter(object):
    def __init__(self, env: gym.Env, agents: list[ACAgent], save_path: str, episode_max_length: int = 500):
        self.env = env
        self.agents = agents
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
            actions = [agent.act(states[idx], training=training) for idx, agent in enumerate(self.agents)]
            # TODO: Add truncated and terminated return vals for old gym envs
            next_states, rewards, dones, info = self.env.step(actions)

            if training:
                for idx, agent in enumerate(self.agents):
                    agent.remember(states[idx], actions[idx], rewards[idx], next_states[idx], int(dones[idx]))

            done = np.all(dones) or episode_length > self.episode_max_length
            states = next_states
            episode_length += 1
            episode_reward += np.sum(rewards)

            if True or render:
                self.env.render()

        return episode_reward, episode_length

    def learn(self, num_steps: int = 50):
        for agent_id, agent in enumerate(self.agents):
            agent.learn(num_steps=num_steps)
            

    def clear_experience(self):
        for agent in self.agents:
            agent.memory.memory.clear()

    def evaluate_policies(self, num_repetitions: int = 10, render: bool = False) -> np.float64:
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
                    
        # save results
        np.save(f"{self.save_path}/experiment_history.npy", self.experiment_history)
        
        


implemented_agent_types = ["IAC", "SNAC", "SEAC"]


def create_experiment(args) -> Experimenter:
    agent_type = args.agent_type
    env_name = args.env
    num_agents = args.num_agents
    episode_max_length = args.episode_max_length
    capacity = args.capacity
    device = args.device
    se_lambda_value = args.SEAC_lambda_value
    save_path = args.save_path
    batch_size = args.batch_size
    
    assert (agent_type in implemented_agent_types)

    env = gym.make(env_name)

    # TODO: Support for multiple lists / teams (e.g. team based where a subset of agents have access to each other)
    # TODO: This is implemented in the 'if agent_type == ...' branches.
    agent_list = []

    # Individual agents with no access to each other
    if agent_type == "IAC":
        for i in range(num_agents):
            agent = ACAgent(env.observation_space[i].shape[0], env.action_space[i].n,
                            capacity=capacity, device=device, batch_size=batch_size)
            
            agent_list.append(agent)

    # Several references to the same agent (shared network)
    elif agent_type == "SNAC":
        agent = ACAgent(env.observation_space[0].shape[0], env.action_space[0].n,
                        capacity=capacity * num_agents, device=device, batch_size=batch_size)
        for i in range(num_agents):
            agent_list.append(agent)

    # Individual agents with access to each other
    elif agent_type == "SEAC":
        for i in range(num_agents):
            agent = SEACAgent(env.observation_space[i].shape[0], env.action_space[i].n,
                              capacity=capacity, device=device,
                              agent_list=agent_list, lambda_value=se_lambda_value, batch_size=batch_size)
            agent_list.append(agent)
            
    if args.pretrain_path is not None:
        for agent in agent_list:
            agent.load(f"{args.pretrain_path}/agent_{agent.agent_id}.pth")

    return Experimenter(env, agent_list, save_path, episode_max_length=episode_max_length)
