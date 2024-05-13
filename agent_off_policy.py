import numpy as np
import torch
import copy

from typing import Tuple, List, Union, Dict, Any

from experience_replay import ExperienceReplay
from nets import ActorPolicyNet, CriticNet, CriticValueNet


class SACAgent(object):
    def __init__(self, obs_shape, action_shape, capacity, device, hidden_size=256, adam_eps=1e-3, gamma=0.99,
                 value_loss_coeff=0.5, learning_rate=3e-4, grad_clip=0.5, tau=8e-3, batch_size=256,
                 n_steps=5, is_discrete=True, n_critics=2, alpha=0.2, auto_alpha=False, value_function_type="Q"):
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.value_loss_coeff = value_loss_coeff
        self.grad_clip = grad_clip
        self.tau = tau
        self.capacity = capacity
        self.n_critics = n_critics
        self.n_steps = n_steps
        
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.value_function_type = value_function_type

        self.memory = ExperienceReplay(capacity)

        self.actor = ActorPolicyNet(obs_shape, action_shape, hidden_size, is_discrete=is_discrete).to(device)
        if value_function_type == "Q":
            self.critics = torch.nn.ModuleList(
                [CriticNet(obs_shape, action_shape if not self.actor.is_discrete else 1, hidden_size).to(device) for _ in
                 range(n_critics)])
        elif value_function_type == "V":
            self.critics = torch.nn.ModuleList(
                [CriticValueNet(obs_shape, hidden_size).to(device) for _ in range(n_critics)])
        else:
            raise ValueError("Invalid value function type. Choose either 'Q' or 'V'.")
        
        self.target_critics = copy.deepcopy(self.critics)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, eps=adam_eps)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=learning_rate, eps=adam_eps)
        
        if self.auto_alpha:
            self.target_entropy = -np.prod(action_shape)
            self.log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate, eps=adam_eps)
            

    def eval(self):
        self.actor.eval()
        self.critics.eval()

    def train(self):
        self.actor.train()
        self.critics.train()

    def soft_update(self):
        for target_param, param in zip(self.target_critics.parameters(), self.critics.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, int(done))

    def remember_tuple(self, transition):
        self.memory.push(*transition)

    def learn(self, num_steps=50):
        if len(self.memory) < self.batch_size:
            return
        for it in range(num_steps):
            states, actions, rewards, next_states, dones = self.memory.sample_tensor(self.batch_size, self.device)

            # Update critics
            with torch.no_grad():
                next_actions_dist = self.actor(next_states)
                if self.actor.is_discrete:
                    next_actions = next_actions_dist.sample()
                    next_log_probs = next_actions_dist.log_prob(next_actions)
                    next_actions, next_log_probs = next_actions.reshape(-1, 1), next_log_probs.reshape(-1, 1)
                else:
                    next_actions = next_actions_dist.rsample()
                    next_log_probs = next_actions_dist.log_prob(next_actions).sum(dim=-1, keepdim=True)

                critic_next_input = torch.cat([next_states, next_actions], -1) if self.value_function_type == "Q" else next_states
                next_qs = torch.stack([critic(critic_next_input) for critic in self.target_critics], dim=0)
                next_q = torch.min(next_qs, dim=0).values
                next_q -= self.alpha * next_log_probs
                target_q = rewards.reshape(-1, 1) + self.gamma * (1 - dones.reshape(-1, 1)) * next_q

            critic_input = torch.cat([states, actions.reshape(self.batch_size, -1)], -1) if self.value_function_type == "Q" else states
            qs = torch.stack([critic(critic_input) for critic in self.critics], dim=0)
            critic_loss = (qs - target_q).pow(2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.grad_clip)
            self.critic_optimizer.step()

            # Update actor
            actions_dist = self.actor(states)
            if self.actor.is_discrete:
                actions = actions_dist.sample()
                log_probs = actions_dist.log_prob(actions)
                actions, log_probs = actions.reshape(-1, 1), log_probs.reshape(-1, 1)
            else:
                actions = actions_dist.rsample()
                log_probs = actions_dist.log_prob(actions).sum(dim=-1, keepdim=True)

            critic_input = torch.cat([states, actions], -1) if self.value_function_type == "Q" else states
            qs = torch.stack([critic(critic_input) for critic in self.critics], dim=0)
            q = torch.min(qs, dim=0).values
            actor_loss = (self.alpha * log_probs - q).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()
            
            if self.auto_alpha:
                alpha_loss = (- self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = torch.clamp(self.log_alpha.detach().exp(), 0, 1).item()

            print(f'Actor Loss: {actor_loss.item():.5f} | Critic Loss: {critic_loss.item():.5f} | Q: {q.mean().item():.5f} | Entropy: {-log_probs.mean().item():.5f}')
            self.soft_update()

    def save(self, path):
        params = {
            'actor': self.actor.state_dict(),
            'critics': self.critics.state_dict()
        }
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(params['actor'])
        self.critics.load_state_dict(params['critics'])
        self.target_critic = copy.deepcopy(self.critic)


# Implement by adding a memory arg which is either None or an Experience Replay Buffer of another agent.
# Capacity should be adjusted accordingly.
class SESACAgent(SACAgent):
    def __init__(self, memory, obs_shape, action_shape, capacity, device, hidden_size=256, adam_eps=1e-3, gamma=0.99,
                 value_loss_coeff=0.5, learning_rate=3e-4, grad_clip=0.5, tau=5e-4, batch_size=256,
                 n_steps=5, is_discrete=True, n_critics=2, alpha=0.2, auto_alpha=False, value_function_type="Q"):
        super(SESACAgent, self).__init__(obs_shape, action_shape, capacity, device, hidden_size, adam_eps, gamma,
                                         value_loss_coeff, learning_rate, grad_clip, tau, batch_size,
                                         n_steps, is_discrete, n_critics, alpha, auto_alpha, value_function_type)
        if memory is not None:
            self.memory = memory
