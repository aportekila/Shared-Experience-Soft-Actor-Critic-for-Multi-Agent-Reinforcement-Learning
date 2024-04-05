import numpy as np
import torch

from experience_replay import ExperienceReplay
from nets import ActorPolicyNet, CriticValueNet


class ACAgent(object):
    def __init__(self, obs_shape, action_shape, capacity, device, hidden_size=64, adam_eps=1e-3, gamma=0.99,
                 entropy_coeff=0.01, value_loss_coeff=0.5, learning_rate=3e-4, grad_clip=0.5, tau=5e-4, batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.grad_clip = grad_clip
        self.tau = tau

        self.memory = ExperienceReplay(capacity)

        self.actor = ActorPolicyNet(obs_shape, action_shape, hidden_size).to(device)
        self.critic = CriticValueNet(obs_shape, hidden_size).to(device)
        self.critic_target = CriticValueNet(obs_shape, hidden_size).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, eps=adam_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, eps=adam_eps)

    def act(self, state: np.ndarray) -> np.integer:
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor(state).mode()
        return action.cpu().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, int(done))

    def learn(self, num_steps=50):
        raise NotImplementedError("Not implemented")

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
        self.critic_target.load_state_dict(params['critic'])

    def soft_update(self):
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class SEACAgent(ACAgent):
    def __init__(self, lambda_value=1.0, **kwargs):
        super(SEACAgent, self).__init__(**kwargs)
        self.lambda_value = lambda_value

    def learn(self, other_memories=None):
        raise NotImplementedError("TODO: Implement the learn method")


class SNACAgent(object):
    def __init__(self, actor, critic, critic_target, actor_optimizer, critic_optimizer, capacity, device, gamma=0.99,
                 entropy_coeff=0.01, value_loss_coeff=0.5, grad_clip=0.5):
        self.device = device
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.grad_clip = grad_clip

        self.memory = ExperienceReplay(capacity)

        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
