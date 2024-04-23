import numpy as np
import torch

from typing import Tuple, List, Union, Dict, Any

from experience_replay import ExperiencePool
from nets import ActorPolicyNet, CriticValueNet


class ACAgent(object):
    def __init__(self, obs_shape, action_shape, device, hidden_size=256, adam_eps=1e-3, gamma=0.99,
                 entropy_coeff=0.01, value_loss_coeff=0.5, learning_rate=3e-4, grad_clip=0.5, tau=5e-4,
                 n_steps=1, is_discrete=True):
        self.device = device
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.grad_clip = grad_clip
        self.tau = tau
        self.n_steps = n_steps

        self.memory = ExperiencePool()

        self.actor = ActorPolicyNet(obs_shape, action_shape, hidden_size, is_discrete=is_discrete).to(device)
        self.critic = CriticValueNet(obs_shape, hidden_size).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, eps=adam_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, eps=adam_eps)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def act(self, state: np.ndarray, training=False) -> np.integer:
        state = torch.FloatTensor(state).to(self.device)
        if training:
            with torch.no_grad():
                action = self.actor.forward(state).sample()
            return action.cpu().numpy()
        else:
            with torch.no_grad():
                action = self.actor(state).mode
            return action.cpu().numpy()
        

    def calculate_loss_terms(self, states, actions, returns) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = states.shape[0]
        dist = self.actor.forward(states)
        if self.actor.is_discrete:
            log_props = dist.log_prob(actions).view(batch_size, -1)
        else:
            log_props = dist.log_prob(actions.clamp(-1 + 1e-6, 1 - 1e-6)).view(batch_size, -1)
        entropy = - log_props # TODO: justify
        state_values = self.critic.forward(states)
        
        # Estimate advantage
        advantages = returns - state_values

        return log_props, entropy, advantages

    def calculate_loss(self, states, actions, returns) -> Tuple[torch.Tensor, torch.Tensor]:
        log_props, entropy, advantages = self.calculate_loss_terms(states, actions, returns)
        # Disregard advantage in gradient calculation for actor
        actor_loss = ((-log_props * advantages.detach()) - entropy * self.entropy_coeff).mean()

        # 'advantages' is just the td errors, take mean squared error.
        critic_loss = self.value_loss_coeff * torch.square(advantages).mean()

        return actor_loss, critic_loss

    def learn(self):
        states, actions, returns = self.memory.sample_tensor(self.device)

        actor_loss, critic_loss = self.calculate_loss(states, actions, returns)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        

    def save(self, path):
        params = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])


class SEACAgent(ACAgent):
    def __init__(self, obs_shape, action_shape, device, agent_list, lambda_value=1.0, **kwargs):
        super(SEACAgent, self).__init__(obs_shape, action_shape, device, **kwargs)
        self.agent_list = agent_list
        self.lambda_value = lambda_value

    def learn(self):
        states, actions, returns = self.memory.sample_tensor(self.device)
        batch_size = states.shape[0]
        
        log_props = self.actor.forward(states).log_prob(actions).view(batch_size, -1)
        
        actor_loss, critic_loss = self.calculate_loss(states, actions, returns)

        for agent in self.agent_list:
            if agent != self:
                states, actions, returns = agent.memory.sample_tensor(self.device)
                log_props_i, _, advantages_i = agent.calculate_loss_terms(states, actions, returns)
                importance_weight = (log_props.exp() / (log_props_i.exp() + 1e-7)).detach()

                actor_loss += self.lambda_value * (importance_weight * -log_props * advantages_i.detach()).mean()
                critic_loss += self.lambda_value * (importance_weight * torch.square(advantages_i)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

