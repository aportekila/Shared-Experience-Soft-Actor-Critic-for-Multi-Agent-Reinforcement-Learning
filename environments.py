from abc import abstractmethod
from typing import Tuple, List

import gymnasium as gym
import rware
from lbforaging.foraging import environment as lbforaging_environment

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
    def __init__(self, **kwargs):
        if kwargs["max_steps"] is None:
            kwargs.pop("max_steps")
        self.env = gym.make('rware-tiny-4ag-v1', **kwargs)
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

        self.action_spaces = {
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


class ForagingEnv(ProjectBaseEnv):
    def __init__(self, **kwargs):
        # Default values
        num_players = kwargs.get("num_players") or 3
        max_level = kwargs.get("max_level") or 3
        field_size = kwargs.get("field_size") or (10, 10)
        max_food = kwargs.get("max_food") or 3
        sight = field_size[0]
        max_steps = kwargs.get("max_steps") or 50
        force_coop = kwargs.get("force_coop") or False

        self.env = lbforaging_environment.ForagingEnv(players=num_players,
                                                      max_player_level=max_level,
                                                      field_size=field_size,
                                                      max_food=max_food,
                                                      sight=sight,
                                                      max_episode_steps=max_steps,
                                                      force_coop=force_coop)
        self.is_discrete = True
        self.possible_agents = [f"agent_{i}" for i in range(num_players)]
        self.timestep = None
        self.max_steps = max_steps

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
            agent: observations_[i * 2] for i, agent in enumerate(self.agents)  # Something weird in how they make obs
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
            observations[agent] = observations_[i * 2]  # same as before
            rewards[agent] = rewards_[i]
            terminateds[agent] = terminateds_[i]
            trancateds[agent] = self.timestep >= self.max_steps
            infos[agent] = {}

        if any(terminateds.values()) or any(trancateds.values()):
            self.agents = []

        return observations, rewards, terminateds, trancateds, infos

    def render(self):
        self.env.render()
