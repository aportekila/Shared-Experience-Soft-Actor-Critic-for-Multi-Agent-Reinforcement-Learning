import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import (
    Normal,
    TransformedDistribution,
    TanhTransform,
    Categorical
)

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


from torch.distributions.transforms import TanhTransform

class SquashedGaussianHead(nn.Module):
    def __init__(self, n, upper_clamp=-2.0):
        super(SquashedGaussianHead, self).__init__()
        self._n = n
        self._upper_clamp = upper_clamp

    def forward(self, x):
        mean_bt = x[..., : self._n]
        log_var_bt = (x[..., self._n :]).clamp(-10, -self._upper_clamp)  # clamp added
        std_bt = log_var_bt.exp().sqrt()
        dist_bt = Normal(mean_bt, std_bt)
        transform = TanhTransform(cache_size=1)
        dist = TransformedDistribution(dist_bt, transform)
        return dist
    
class ActorPolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, is_discrete=True):
        super(ActorPolicyNet, self).__init__()

        self.is_discrete = is_discrete
        self.arch = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size if is_discrete else 2 *action_size)
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.arch:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x) -> torch.distributions.Distribution:
        x = self.arch(x)
        if self.is_discrete:
            return Categorical(logits=x)
        else:
            return SquashedGaussianHead(x.size(-1) // 2)(x)
