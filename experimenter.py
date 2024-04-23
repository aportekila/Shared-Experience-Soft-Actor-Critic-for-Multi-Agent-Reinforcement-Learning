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
import time
from typing import List

from environments import make_envs, ProjectBaseEnv
from agent import ACAgent, SEACAgent
from experience_replay import NStepExperienceReplay
from utils import seed_everything



class Experimenter(object):
    def __init__(self, args, envs: List[ProjectBaseEnv], eval_env: ProjectBaseEnv, agents: list[ACAgent], save_path: str):
        self.args = args
        self.envs = envs
        self.eval_env = eval_env
        self.learning_agents = agents
        self.agent_names = self.eval_env.agents
        self.save_path = save_path
        self.n_agents = len(self.agent_names)
        self.experiment_history = {
            "timestep": [],
            "mean_reward": [],
            "std_reward": [],
            "mean_length": [],
            "std_length": []
        }

        self.temp_memories = []  # for n-step TD learning
        
    def run(self, args):
        start = time.time()
        
        buffers = [{agent_id: NStepExperienceReplay() for agent_id in self.agent_names} for _ in range(args.num_processes)]
        states = [env.reset()[0] for env in self.envs] # exclude info
        
        n_interactions = args.n_steps * args.num_processes
        num_updates = int(args.total_env_steps // n_interactions)
        
        
        
        for update in tqdm(range(1, num_updates + 1)):
            
            # for each environment
            for p_id, (state, env) in enumerate(zip(states, self.envs)):
                
                # n-step TD learning
                for n_step in range(args.n_steps):
                    actions = {self.agent_names[agent_id]: agent.act(state[self.agent_names[agent_id]], training=True) for agent_id, agent in enumerate(self.learning_agents)}
                    next_states, rewards, terminals, truncateds, _ = env.step(list(actions.values()))
                    done = np.all(list(terminals.values())) or np.all(list(truncateds.values()))
                    
                    for agent_id in self.agent_names:
                        buffers[p_id][agent_id].push(state[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id], done * 1.0)
                    
                    state = next_states
                    if done: # reset environment
                        state, info = env.reset()
                
                states[p_id] = state
                
            for agent_id, agent in enumerate(self.learning_agents):
                transitions = {"states": [], "actions": [], "returns": []}
                for env_id in range(args.num_processes):
                    last_state = buffers[env_id][self.agent_names[agent_id]].memory['next_state'][-1]
                    with torch.no_grad():
                        last_value = agent.critic(torch.tensor(last_state, device=agent.device, dtype=torch.float32)).item()
                    temp_transitions = buffers[env_id][self.agent_names[agent_id]].convert_to_n_step(last_value, agent.gamma)
                    transitions["states"].extend(temp_transitions["state"])
                    transitions["actions"].extend(temp_transitions["action"])
                    transitions["returns"].extend(temp_transitions["return"])
                
                agent.memory.insert_transitions(transitions)
                
            for agent_id, agent in enumerate(self.learning_agents):
                agent.learn()
                
            # clear buffers
            for buffer in buffers:
                for agent_id in self.agent_names:
                    buffer[agent_id].clear()
            
            # clear memories
            for agent in self.learning_agents:
                agent.memory.clear()

            if (update - 1) % args.evaluate_frequency == 0 or update == num_updates:
                timestep = update * n_interactions
                self.evaluate_and_save(args, timestep)
                
        end = time.time()
        print(f"Training took {end - start:.2f} seconds")
            
    
    def evaluate_and_save(self, args, timestep):
        result = self.evaluate_policies(args.evaluate_episodes, render=args.render)
        self.experiment_history["timestep"].append(timestep)
        for key, value in result.items():
            print(f"{key}: {value:.5f}", end=" ")
            self.experiment_history[key].append(value)
        print()
        for agent_id, agent in enumerate(self.learning_agents):
            agent.save(f"{self.save_path}/agent_{agent_id}.pth")

        # save results periodically
        mean_rewards = np.array(self.experiment_history["mean_reward"])
        std_rewards = np.array(self.experiment_history["std_reward"])
        x_axis = np.array(self.experiment_history["timestep"])
        plt.plot(x_axis, mean_rewards)
        plt.fill_between(x_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
        plt.xlabel("Time Steps")
        plt.ylabel("Episode Rewards")
        plt.title(f"{args.env} - {args.agent_type}")
        plt.savefig(f"{self.save_path}/results.svg")
        plt.close()
        np.save(f"{self.save_path}/experiment_history.npy", self.experiment_history)

    def evaluate_policies(self, num_repetitions: int = 10, render: bool = False) -> dict:
        for agent in self.learning_agents:
            agent.eval()

        rewards, lengths = [], []
        for i in range(num_repetitions):
            episode_reward, episode_length = self.evaluate_episode(render=(i == 0 and render))
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
                        

    def evaluate_episode(self, render: bool = False) -> tuple[np.float64, int]:
        states, info = self.eval_env.reset()
        episode_reward = 0
        episode_length = 0
        # This is pettingzoo convention. List will be empty when done.
        while self.eval_env.agents:
            # States is a dictionary with agent name as key
            actions = {agent_id: agent.act(states[agent_id], training=False)
                       for agent, agent_id in zip(self.learning_agents, self.agent_names)}

            next_states, rewards, terminated, truncated, info = self.eval_env.step(list(actions.values()))

            states = next_states
            episode_length += 1
            episode_reward += np.sum(list(rewards.values()))

            if render:
                self.eval_env.render()
                
        return episode_reward, episode_length

implemented_agent_types = ["IAC", "SNAC", "SEAC"]


def create_experiment(args) -> Experimenter:
    agent_type: str = args.agent_type
    env_name: str = args.env
    device: torch.device = args.device
    se_lambda_value: float = args.SEAC_lambda_value
    save_path: str = args.save_path
    episode_max_length: int = args.episode_max_length
    n_steps: int = args.n_steps
    num_processes: int = args.num_processes
    seed: int = args.seed

    assert (agent_type in implemented_agent_types)

    envs, eval_env = make_envs(env_name=env_name, num_processes=num_processes, max_steps=episode_max_length)

    is_discrete = eval_env.is_discrete
    # TODO: Support for multiple lists / teams (e.g. team based where a subset of agents have access to each other)
    agent_list = []

    for rank, env in enumerate(envs):
        env.reset(seed=seed + rank + 1)
    eval_env.reset(seed=seed)
        
    # Set seed for reproducibility
    seed_everything(seed)

    # Individual agents with no access to each other
    if agent_type == "IAC":
        for agent in eval_env.agents:
            agent = ACAgent(eval_env.observation_shapes[agent], eval_env.action_shapes[agent],
                            device=device, n_steps=n_steps, is_discrete=is_discrete)
            agent_list.append(agent)

    # Several references to the same agent (shared network)
    elif agent_type == "SNAC":
        num_agents = len(eval_env.agents)
        #  TODO: update agent.py
        agent = ACAgent(eval_env.observation_shapes[eval_env.agents[0]], eval_env.action_shapes[eval_env.agents[0]],
                        device=device, n_steps=n_steps, is_discrete=is_discrete)
        for i in range(num_agents):
            agent_list.append(agent)

    # Individual agents with access to each other
    elif agent_type == "SEAC":
        for agent in eval_env.agents:
            agent = SEACAgent(eval_env.observation_shapes[agent], eval_env.action_shapes[agent],
                              device=device, agent_list=agent_list, lambda_value=se_lambda_value, 
                              n_steps=n_steps, is_discrete=is_discrete)
            agent_list.append(agent)

    if args.pretrain_path is not None:
        for agent_id, agent in enumerate(agent_list):
            agent.load(f"{args.pretrain_path}/agent_{agent_id}.pth")

    return Experimenter(args, envs, eval_env, agent_list, save_path)
