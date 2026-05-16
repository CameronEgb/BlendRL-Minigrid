import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLP(nn.Module):
    def __init__(self, device, hidden_sizes=[64, 64], has_softmax=False, has_sigmoid=False, out_size=2, as_dict=False, logic=False):
        super().__init__()
        self.device = device
        self.logic = logic
        self.num_in_features = 4 if not logic else 8

        # Backbone: Feature extraction
        layers = []
        last_size = self.num_in_features
        for size in hidden_sizes:
            layers.append(layer_init(nn.Linear(last_size, size)))
            layers.append(nn.Tanh())
            last_size = size
        self.network = nn.Sequential(*layers)
        
        # Heads: Standardized names
        self.actor = layer_init(nn.Linear(last_size, out_size), std=0.01 if not logic else 1.0)
        self.critic = layer_init(nn.Linear(last_size, 1), std=1.0)

        # Optional activation layers
        self.softmax = nn.Softmax(dim=-1) if has_softmax else nn.Identity()
        self.sigmoid = nn.Sigmoid() if has_sigmoid else nn.Identity()

        self.to(device)

    def forward(self, x):
        x = x.float().reshape(x.shape[0], -1)
        hidden = self.network(x)
        logits = self.actor(hidden)
        return self.softmax(logits)

    def get_value(self, x, logic_state=None):
        x = x.float().reshape(x.shape[0], -1)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        x = x.float().reshape(x.shape[0], -1)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def act(self, x, logic_state=None, epsilon=0.0):
        # Compatibility with Renderer and wrappers
        x = x.float().reshape(x.shape[0], -1)
        action_probs = self.forward(x)
        dist = Categorical(probs=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action, action_logprob

    def _print(self):
        return "Neural Agent (MLP) - No logic rules."
