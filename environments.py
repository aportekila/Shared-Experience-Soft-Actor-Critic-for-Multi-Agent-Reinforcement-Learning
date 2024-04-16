from abc import abstractmethod
from typing import Tuple, List

import gymnasium as gym
import rware
import lbforaging

from copy import copy
from pettingzoo import ParallelEnv


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


class RwareEnvironment(ProjectBaseEnv):
    def __init__(self, env_name='rware-tiny-4ag-v1', **kwargs):
        if kwargs["max_steps"] is None:
            kwargs.pop("max_steps")
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

        return observations, rewards, terminateds, trancateds, infos

    def render(self):
        self.env.render()


class ForagingEnvironment(RwareEnvironment):
    def __init__(self, env_name='Foraging-10x10-3p-3f-v2', **kwargs):
        super().__init__(env_name, **kwargs)

