import torch

from experience_replay import ExperienceReplay
from nets import ActorPolicyNet, CriticValueNet


class ACAgent(object):
    def __init__(self, obs_shape, action_shape, capacity, device, hidden_size=64, adam_eps=1e-3, gamma=0.99,
                 entropy_coeff=0.01, value_loss_coeff=0.5, learning_rate=3e-4, grad_clip=0.5):
        self.device = device
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.grad_clip = grad_clip

        self.memory = ExperienceReplay(capacity)

        self.actor = ActorPolicyNet(obs_shape, action_shape, hidden_size).to(device)
        self.critic = CriticValueNet(obs_shape, hidden_size).to(device)
        self.critic_target = CriticValueNet(obs_shape, hidden_size).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, eps=adam_eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, eps=adam_eps)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        raise NotImplementedError("TODO: Implement the learn method")

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
