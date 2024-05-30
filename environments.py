from abc import abstractmethod
from typing import Tuple, List

import gymnasium as gym
import numpy as np
import rware
import lbforaging

from copy import copy
from pettingzoo import ParallelEnv
from pettingzoo.sisl import multiwalker_v9
from pettingzoo.sisl import waterworld_v4


class ProjectBaseEnv(ParallelEnv):
    # I think this abstractmethod thing helps IDE not complain
    @abstractmethod
    def __init__(self, **kwargs):
        self.env: gym.Env = None
        self.is_discrete: bool = None
        self.possible_agents: List[str] = None
        self.timestep: int = None
        self.max_steps: int = None
        self.observation_spaces: dict = None
        self.action_spaces: dict = None
        self.observation_shapes: dict = None
        self.action_spaces: dict = None

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        raise NotImplementedError("Must be implemented in subclass")

    def step(self, actions: list) -> Tuple[dict, dict, dict, dict, dict]:
        raise NotImplementedError("Must be implemented in subclass")

    def render(self):
        raise NotImplementedError("Must be implemented in subclass")


class MountainCarEnvironment(ProjectBaseEnv):
    def __init__(self, **kwargs):
        self.env = gym.make('MountainCarContinuous-v0')
        self.is_discrete = False
        self.possible_agents = [f"agent"]
        self.timestep = None
        self.max_steps = 999

        self.observation_spaces = {
            "agent": self.env.observation_space
        }

        self.action_spaces = {
            "agent": self.env.action_space
        }

        self.observation_shapes = {
            "agent": 2
        }

        self.action_shapes = {
            "agent": 1
        }

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        observation, info = self.env.reset(seed=seed, options=options)
        observations = {
            "agent": observation
        }
        infos = {
            "agent": {}
        }

        return observations, infos

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions[0])
        self.timestep += 1
        observations = {"agent": observation}
        rewards = {"agent": reward}
        terminateds = {"agent": terminated}
        trancateds = {"agent": truncated}
        infos = {"agent": {}}

        if any(terminateds.values()) or any(trancateds.values()):
            self.agents = []

        return observations, rewards, terminateds, trancateds, infos

    def render(self):
        self.env.render()


class PendulumEnvironment(ProjectBaseEnv):
    def __init__(self, **kwargs):
        self.env = gym.make('Pendulum-v1')
        self.is_discrete = False
        self.possible_agents = [f"agent"]
        self.timestep = None
        self.max_steps = 200

        self.observation_spaces = {
            "agent": self.env.observation_space
        }

        self.action_spaces = {
            "agent": self.env.action_space
        }

        self.observation_shapes = {
            "agent": 3
        }

        self.action_shapes = {
            "agent": 1
        }

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        observation, info = self.env.reset(seed=seed, options=options)
        observations = {
            "agent": observation
        }
        infos = {
            "agent": {}
        }

        return observations, infos

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions[0])
        self.timestep += 1
        observations = {"agent": observation}
        rewards = {"agent": reward}
        terminateds = {"agent": terminated}
        trancateds = {"agent": truncated}
        infos = {"agent": {}}

        if any(terminateds.values()) or any(trancateds.values()):
            self.agents = []

        return observations, rewards, terminateds, trancateds, infos

    def render(self):
        self.env.render()


class RwareEnvironment(ProjectBaseEnv):
    def __init__(self, env_name='rware-tiny-4ag-v1', **kwargs):
        if kwargs["max_steps"] is None:
            kwargs.pop("max_steps")

        render = kwargs.pop("render") or False
        self.should_render = render
        self.env = gym.make(env_name, **kwargs)
        self.is_discrete = True
        self.possible_agents = [f"agent_{i}" for i in range(self.env.n_agents)]
        self.timestep = None
        self.max_steps = self.env.unwrapped.max_steps

        self.observation_spaces = {
            agent: self.env.observation_space[i] for i, agent in enumerate(self.possible_agents)
        }

        self.action_spaces = {
            agent: self.env.action_space[i] for i, agent in enumerate(self.possible_agents)
        }

        self.observation_shapes = {
            agent: self.observation_spaces[agent].shape[0] for agent in self.possible_agents
        }

        self.action_shapes = {
            agent: self.action_spaces[agent].n for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        observations_, infos_ = self.env.reset(seed=seed, options=options)
        observations = {
            agent: observations_[i] for i, agent in enumerate(self.agents)
        }
        infos = {
            agent: {} for agent in self.agents
        }

        return observations, infos

    def step(self, actions):
        observations_, rewards_, terminateds_, truncateds_, infos_ = self.env.step(actions)
        self.timestep += 1
        observations = {}
        rewards = {}
        terminateds = {}
        trancateds = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = observations_[i]
            rewards[agent] = rewards_[i]
            terminateds[agent] = terminateds_[i]
            trancateds[agent] = self.timestep >= self.max_steps
            infos[agent] = {}

        if any(terminateds.values()) or any(trancateds.values()):
            self.agents = []

        if self.should_render:
            self.render()

        return observations, rewards, terminateds, trancateds, infos

    def render(self):
        self.env.render()


class ForagingEnvironment(ProjectBaseEnv):

    def __init__(self, env_name='Foraging-10x10-3p-3f-v2', **kwargs):
        if kwargs["max_steps"] is None:
            kwargs.pop("max_steps")
        render = False #kwargs.pop("render") or False
        self.should_render = render
        self.env = gym.make(env_name, **kwargs)
        self.is_discrete = True
        self.possible_agents = [f"agent_{i}" for i in range(self.env.n_agents)]
        self.timestep = None
        self.max_steps = self.env.unwrapped._max_episode_steps

        self.observation_spaces = {
            agent: self.env.observation_space[i] for i, agent in enumerate(self.possible_agents)
        }

        self.action_spaces = {
            agent: self.env.action_space[i] for i, agent in enumerate(self.possible_agents)
        }

        self.observation_shapes = {
            agent: self.observation_spaces[agent].shape[0] for agent in self.possible_agents
        }

        self.action_shapes = {
            agent: self.action_spaces[agent].n for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        observations_, infos_ = self.env.reset(seed=seed, options=options)
        observations = {
            agent: observations_[i] for i, agent in enumerate(self.agents)
        }
        infos = {
            agent: {} for agent in self.agents
        }

        return observations, infos

    def step(self, actions):
        observations_, rewards_, terminateds_, truncateds_, infos_ = self.env.step(actions)
        self.timestep += 1
        observations = {}
        rewards = {}
        terminateds = {}
        trancateds = {}
        infos = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = observations_[i]
            rewards[agent] = rewards_[i]
            terminateds[agent] = terminateds_[i]
            trancateds[agent] = self.timestep >= self.max_steps
            infos[agent] = {}

        if any(terminateds.values()) or any(trancateds.values()):
            self.agents = []

        if self.should_render:
            self.render()

        return observations, rewards, terminateds, trancateds, infos

    def render(self):
        self.env.render()


class PettingZooEnvironment(ProjectBaseEnv):
    def __init__(self, **kwargs):
        env_name = kwargs.pop("env_name")
        max_steps = kwargs.pop("max_steps") or 500
        render = False #kwargs.pop("render") or False
        render_mode = "human" if render else None
        if env_name == "multiwalker":
            self.env = multiwalker_v9.parallel_env(max_cycles=max_steps, shared_reward=False, render_mode=render_mode)
        elif env_name == "waterworld":
            self.env = waterworld_v4.parallel_env(max_cycles=max_steps, render_mode=render_mode)
        else:
            raise ValueError("Invalid environment name")
        self.env.reset()
        self.is_discrete = False
        self.possible_agents = self.env.possible_agents
        self.timestep = None
        self.max_steps = max_steps

        self.observation_spaces = {
            agent: self.env.observation_space(agent) for agent in self.env.agents
        }

        self.action_spaces = {
            agent: self.env.action_space(agent) for agent in self.env.agents
        }

        self.observation_shapes = {
            agent: self.observation_spaces[agent].shape[0] for agent in self.possible_agents
        }

        self.action_shapes = {
            agent: self.action_spaces[agent].shape[0] for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        obs, infos = self.env.reset(seed=seed, options=options)
        self.timestep = 0
        self.agents = self.env.agents
        return obs, infos

    def step(self, actions: list) -> Tuple[dict, dict, dict, dict, dict]:
        # TODO: This should be given as a dictionary, and we should change the implementation elsewhere.
        actions = {agent: actions[i] for i, agent in enumerate(self.agents)}

        observations, rewards, terminates, truncates, infos = self.env.step(actions)
        self.timestep += 1
        self.agents = self.env.agents
        return observations, rewards, terminates, truncates, infos

    def render(self):
        self.env.render()
