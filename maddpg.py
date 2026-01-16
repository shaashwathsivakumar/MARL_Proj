"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm.
"""
import os
import pickle
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import Adam

from networks import Actor, Critic
from buffer import MultiAgentReplayBuffer


class MADDPGAgent:
    """
    Single agent within MADDPG framework.
    Contains actor, critic, and their target networks.
    """
    def __init__(self, obs_dim, action_dim, critic_input_dim, actor_lr, critic_lr, device='cpu'):
        self.device = device

        # Actor: maps own observation to action
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.target_actor = deepcopy(self.actor)

        # Critic: maps all observations + all actions to Q-value
        self.critic = Critic(critic_input_dim).to(device)
        self.target_critic = deepcopy(self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs, explore=True):
        """
        Select action given observation.

        Args:
            obs: Numpy array of observation
            explore: Whether to use stochastic action selection

        Returns:
            Integer action index
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor.get_action(obs_tensor, explore=explore)
        return action.squeeze(0).argmax().item()

    def get_target_action(self, obs_batch):
        """Get actions from target actor for a batch of observations."""
        # Use Gumbel-Softmax for target actions (matches reference implementation)
        logits = self.target_actor(obs_batch)
        return F.gumbel_softmax(logits, hard=True).detach()


class MADDPG:
    """
    MADDPG algorithm managing multiple agents.

    Key features:
    - Centralized training with decentralized execution
    - Each agent's critic sees all observations and actions
    - Gumbel-Softmax for differentiable discrete actions
    - Optional: geometric sampling for recent experience bias
    - Optional: previous action conditioning for critic
    """
    def __init__(self, agent_ids, obs_dims, action_dims, buffer_capacity,
                 actor_lr=0.01, critic_lr=0.01, device='cpu',
                 geometric_sampling=False, geo_alpha=1e-5, use_prev_action=False):
        """
        Args:
            agent_ids: List of agent identifiers
            obs_dims: Dict mapping agent_id -> observation dimension
            action_dims: Dict mapping agent_id -> action dimension
            buffer_capacity: Size of replay buffer
            actor_lr: Learning rate for actor networks
            critic_lr: Learning rate for critic networks
            device: Torch device
            geometric_sampling: Whether to use geometric sampling in replay buffer
            geo_alpha: Decay rate for geometric sampling
            use_prev_action: Whether to condition critic on previous joint action
        """
        self.agent_ids = agent_ids
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.device = device
        self.use_prev_action = use_prev_action

        # Critic input: all observations + all actions (+ previous actions if enabled)
        total_obs_dim = sum(obs_dims.values())
        total_action_dim = sum(action_dims.values())
        self.total_action_dim = total_action_dim

        critic_input_dim = total_obs_dim + total_action_dim
        if use_prev_action:
            critic_input_dim += total_action_dim  # Add space for previous joint action

        # Create agents
        self.agents = {
            agent_id: MADDPGAgent(
                obs_dims[agent_id],
                action_dims[agent_id],
                critic_input_dim,
                actor_lr,
                critic_lr,
                device
            )
            for agent_id in agent_ids
        }

        # Shared replay buffer with per-agent storage
        self.buffer = MultiAgentReplayBuffer(
            agent_ids, buffer_capacity, obs_dims, action_dims, device,
            geometric_sampling, geo_alpha, use_prev_action
        )

    def select_actions(self, observations, explore=True):
        """
        Select actions for all agents.

        Args:
            observations: Dict mapping agent_id -> observation

        Returns:
            Dict mapping agent_id -> action index
        """
        return {
            agent_id: self.agents[agent_id].select_action(observations[agent_id], explore)
            for agent_id in self.agent_ids
        }

    def store_transition(self, observations, actions, rewards, next_observations, dones,
                         prev_joint_action=None):
        """Store a transition for all agents."""
        self.buffer.add(observations, actions, rewards, next_observations, dones,
                        prev_joint_action)

    def update(self, batch_size, gamma):
        """
        Perform one update step for all agents.

        Args:
            batch_size: Number of transitions to sample
            gamma: Discount factor
        """
        batch = self.buffer.sample(batch_size)

        # Compute target actions for all agents using target networks
        target_actions = {
            agent_id: self.agents[agent_id].get_target_action(batch['next_obs'][agent_id])
            for agent_id in self.agent_ids
        }

        for agent_id in self.agent_ids:
            self._update_agent(agent_id, batch, target_actions, gamma)

    def _update_agent(self, agent_id, batch, target_actions, gamma):
        """Update a single agent's actor and critic."""
        agent = self.agents[agent_id]

        # === Critic Update ===
        # Current Q-value: Q(s, a) where s and a are all agents' obs/actions
        current_obs = torch.cat([batch['obs'][aid] for aid in self.agent_ids], dim=1)
        current_actions = torch.cat([batch['actions'][aid] for aid in self.agent_ids], dim=1)

        # Build critic input (optionally include previous joint action)
        if self.use_prev_action:
            prev_joint_action = batch['prev_joint_action']
            critic_input = torch.cat([current_obs, prev_joint_action, current_actions], dim=1)
        else:
            critic_input = torch.cat([current_obs, current_actions], dim=1)

        current_q = agent.critic(critic_input).squeeze(1)

        # Target Q-value: r + gamma * Q'(s', a') * (1 - done)
        next_obs = torch.cat([batch['next_obs'][aid] for aid in self.agent_ids], dim=1)
        next_actions = torch.cat([target_actions[aid] for aid in self.agent_ids], dim=1)

        with torch.no_grad():
            # For target: prev_action at next state = current_actions
            if self.use_prev_action:
                target_critic_input = torch.cat([next_obs, current_actions, next_actions], dim=1)
            else:
                target_critic_input = torch.cat([next_obs, next_actions], dim=1)

            target_q = agent.target_critic(target_critic_input).squeeze(1)
            td_target = batch['rewards'][agent_id] + gamma * target_q * (1 - batch['dones'][agent_id])

        critic_loss = F.mse_loss(current_q, td_target)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # === Actor Update ===
        # Get fresh action from current agent's actor (with gradients)
        # Other agents use their stored actions from the buffer (matches reference)
        action, current_logits = self.agents[agent_id].actor.get_action(
            batch['obs'][agent_id], explore=True, return_logits=True
        )

        # Build action list: current agent uses fresh action, others use buffer actions
        actor_actions = []
        for aid in self.agent_ids:
            if aid == agent_id:
                actor_actions.append(action)
            else:
                actor_actions.append(batch['actions'][aid])

        # Actor loss: maximize Q-value
        all_actor_actions = torch.cat(actor_actions, dim=1)

        if self.use_prev_action:
            actor_critic_input = torch.cat([current_obs, prev_joint_action, all_actor_actions], dim=1)
        else:
            actor_critic_input = torch.cat([current_obs, all_actor_actions], dim=1)

        actor_loss = -agent.critic(actor_critic_input).mean()

        # Regularization: penalize large logits
        reg_loss = (current_logits ** 2).mean()

        total_actor_loss = actor_loss + 1e-3 * reg_loss

        agent.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()

    def soft_update_targets(self, tau):
        """Soft update all target networks."""
        for agent in self.agents.values():
            self._soft_update(agent.target_actor, agent.actor, tau)
            self._soft_update(agent.target_critic, agent.critic, tau)

    @staticmethod
    def _soft_update(target, source, tau):
        """Soft update: target = tau * source + (1 - tau) * target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

    def save(self, path):
        """Save model weights."""
        state = {
            agent_id: {
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()
            }
            for agent_id, agent in self.agents.items()
        }
        torch.save(state, os.path.join(path, 'model.pt'))

    def load(self, path):
        """Load model weights."""
        state = torch.load(os.path.join(path, 'model.pt'), map_location=self.device)
        for agent_id, agent in self.agents.items():
            agent.actor.load_state_dict(state[agent_id]['actor'])
            agent.critic.load_state_dict(state[agent_id]['critic'])
            agent.target_actor = deepcopy(agent.actor)
            agent.target_critic = deepcopy(agent.critic)
