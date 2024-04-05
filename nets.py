import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticValueNet(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(CriticValueNet, self).__init__()

        self.arch = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.arch:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.arch(x)


#  TODO: Check the following code
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        #  TODO: Check the following code
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class ActorPolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, is_discrete=True):
        super(ActorPolicyNet, self).__init__()

        self.is_discrete = is_discrete
        self.arch = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.arch:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.arch(x)
        if self.is_discrete:
            return FixedCategorical(logits=x)
        else:
            raise NotImplementedError('ActorPolicyNet only supports discrete actions')
