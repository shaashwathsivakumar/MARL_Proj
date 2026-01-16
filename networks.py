"""
Neural network architectures for MADDPG.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    """Xavier uniform initialization for linear layers."""
    if isinstance(module, nn.Linear):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.01)


class Actor(nn.Module):
    """
    Actor network that maps observations to action probabilities.
    Uses Gumbel-Softmax for differentiable discrete action sampling.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.net.apply(init_weights)

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, obs, explore=True, return_logits=False):
        """
        Get action using Gumbel-Softmax for differentiable sampling.

        Args:
            obs: Observation tensor
            explore: If True, use Gumbel-Softmax sampling; if False, use argmax
            return_logits: If True, also return raw logits

        Returns:
            action: One-hot action tensor (differentiable if explore=True)
            logits: Raw network output (only if return_logits=True)
        """
        logits = self.forward(obs)

        if explore:
            # Gumbel-Softmax provides differentiable sampling from categorical
            action = F.gumbel_softmax(logits, hard=True)
        else:
            # Deterministic action selection
            action = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).float()

        if return_logits:
            return action, logits
        return action


class Critic(nn.Module):
    """
    Critic network that estimates Q-value given all observations and actions.
    Input: concatenation of all agents' observations and actions.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.net.apply(init_weights)

    def forward(self, obs_and_actions):
        """
        Args:
            obs_and_actions: Concatenated tensor of all observations and actions

        Returns:
            Q-value estimate
        """
        return self.net(obs_and_actions)
