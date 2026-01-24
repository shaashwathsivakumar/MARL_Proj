"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm.
"""
import os
import pickle
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from networks import Actor, Critic
from buffer import MultiAgentReplayBuffer


# Agent team definitions for shared actor grouping
# Maps environment name to function that returns list of "agent" team members
AGENT_TEAMS = {
    'simple_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_adversary_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_crypto_v3': lambda agents: [a for a in agents if a.startswith('alice') or a.startswith('bob')],
    'simple_push_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_reference_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_speaker_listener_v4': lambda agents: agents,  # All agents on same team
    'simple_spread_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_tag_v3': lambda agents: [a for a in agents if a.startswith('agent')],
    'simple_world_comm_v3': lambda agents: [a for a in agents if a.startswith('agent')],
}

# Environments incompatible with shared actors due to heterogeneous dimensions
INCOMPATIBLE_SHARED_ACTOR_ENVS = ['simple_speaker_listener_v4', 'simple_world_comm_v3']


def get_agent_team(env_name, agent_ids):
    """Get list of agents in the 'agent' team (non-adversaries)."""
    if env_name in AGENT_TEAMS:
        return AGENT_TEAMS[env_name](agent_ids)
    return [a for a in agent_ids if a.startswith('agent')]


class MADDPGAgent:
    """
    Single agent within MADDPG framework.
    Contains actor, critic, and their target networks.

    When shared_actor is provided, the actor and target_actor are shared
    across multiple agents in a team, with the optimizer managed externally.
    """
    def __init__(self, obs_dim, action_dim, critic_input_dim, actor_lr, critic_lr, device='cpu',
                 shared_actor=None, shared_target_actor=None):
        self.device = device
        self.shared_actor_mode = shared_actor is not None

        # Actor: use shared or create own
        if shared_actor is not None:
            self.actor = shared_actor
            self.target_actor = shared_target_actor
            self.actor_optimizer = None  # Optimizer managed externally
        else:
            self.actor = Actor(obs_dim, action_dim).to(device)
            self.target_actor = deepcopy(self.actor)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)

        # Critic: always individual (centralized training)
        self.critic = Critic(critic_input_dim).to(device)
        self.target_critic = deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs, prev_obs=None, explore=True):
        """
        Select action given observation.

        Args:
            obs: Numpy array of observation
            prev_obs: Optional previous observation for temporal context
            explore: Whether to use stochastic action selection

        Returns:
            Integer action index
        """
        # Concatenate previous observation if provided
        if prev_obs is not None:
            obs = np.concatenate([prev_obs, obs])
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
    - Optional: previous observation conditioning for actor
    - Optional: shared actor networks within teams
    """
    def __init__(self, agent_ids, obs_dims, action_dims, buffer_capacity,
                 actor_lr=0.01, critic_lr=0.01, device='cpu',
                 geometric_sampling=False, geo_alpha=1e-5, use_prev_action=False,
                 use_prev_obs=False, shared_actor=False, env_name=None):
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
            use_prev_obs: Whether to condition actor on previous observation
            shared_actor: Whether to share actor networks within teams
            env_name: Environment name (required if shared_actor=True for team detection)
        """
        self.agent_ids = agent_ids
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.device = device
        self.use_prev_action = use_prev_action
        self.use_prev_obs = use_prev_obs

        # Handle shared_actor flag
        self.shared_actor = shared_actor
        if shared_actor and env_name in INCOMPATIBLE_SHARED_ACTOR_ENVS:
            print(f"Warning: {env_name} has heterogeneous agent dimensions. "
                  f"Falling back to individual actors.")
            self.shared_actor = False

        # Critic input: all observations + all actions (+ previous actions if enabled)
        total_obs_dim = sum(obs_dims.values())
        total_action_dim = sum(action_dims.values())
        self.total_action_dim = total_action_dim

        critic_input_dim = total_obs_dim + total_action_dim
        if use_prev_action:
            critic_input_dim += total_action_dim  # Add space for previous joint action

        # Setup shared actors if enabled
        if self.shared_actor:
            # Determine team membership
            self.agent_team = get_agent_team(env_name, agent_ids) if env_name else agent_ids
            self.adversary_team = [a for a in agent_ids if a not in self.agent_team]

            # Create shared actor for agent team
            agent_obs_dim = obs_dims[self.agent_team[0]] * 2 if use_prev_obs else obs_dims[self.agent_team[0]]
            agent_action_dim = action_dims[self.agent_team[0]]
            self.shared_agent_actor = Actor(agent_obs_dim, agent_action_dim).to(device)
            self.shared_agent_target_actor = deepcopy(self.shared_agent_actor)
            self.shared_agent_actor_optimizer = Adam(self.shared_agent_actor.parameters(), lr=actor_lr)

            # Create shared actor for adversary team (if any)
            if self.adversary_team:
                adv_obs_dim = obs_dims[self.adversary_team[0]] * 2 if use_prev_obs else obs_dims[self.adversary_team[0]]
                adv_action_dim = action_dims[self.adversary_team[0]]
                self.shared_adversary_actor = Actor(adv_obs_dim, adv_action_dim).to(device)
                self.shared_adversary_target_actor = deepcopy(self.shared_adversary_actor)
                self.shared_adversary_actor_optimizer = Adam(self.shared_adversary_actor.parameters(), lr=actor_lr)
            else:
                self.shared_adversary_actor = None
                self.shared_adversary_target_actor = None
                self.shared_adversary_actor_optimizer = None
        else:
            self.agent_team = None
            self.adversary_team = None

        # Create agents
        self.agents = {}
        for agent_id in agent_ids:
            actor_obs_dim = obs_dims[agent_id] * 2 if use_prev_obs else obs_dims[agent_id]

            if self.shared_actor:
                # Use shared actors
                if agent_id in self.agent_team:
                    shared_actor_net = self.shared_agent_actor
                    shared_target_actor_net = self.shared_agent_target_actor
                else:
                    shared_actor_net = self.shared_adversary_actor
                    shared_target_actor_net = self.shared_adversary_target_actor

                self.agents[agent_id] = MADDPGAgent(
                    actor_obs_dim,
                    action_dims[agent_id],
                    critic_input_dim,
                    actor_lr,
                    critic_lr,
                    device,
                    shared_actor=shared_actor_net,
                    shared_target_actor=shared_target_actor_net
                )
            else:
                # Individual actors (original behavior)
                self.agents[agent_id] = MADDPGAgent(
                    actor_obs_dim,
                    action_dims[agent_id],
                    critic_input_dim,
                    actor_lr,
                    critic_lr,
                    device
                )

        # Shared replay buffer with per-agent storage
        self.buffer = MultiAgentReplayBuffer(
            agent_ids, buffer_capacity, obs_dims, action_dims, device,
            geometric_sampling, geo_alpha, use_prev_action, use_prev_obs
        )

    def select_actions(self, observations, prev_observations=None, explore=True):
        """
        Select actions for all agents.

        Args:
            observations: Dict mapping agent_id -> observation
            prev_observations: Optional dict mapping agent_id -> previous observation
            explore: Whether to use exploration

        Returns:
            Dict mapping agent_id -> action index
        """
        return {
            agent_id: self.agents[agent_id].select_action(
                observations[agent_id],
                prev_observations[agent_id] if prev_observations else None,
                explore
            )
            for agent_id in self.agent_ids
        }

    def store_transition(self, observations, actions, rewards, next_observations, dones,
                         prev_joint_action=None, prev_observations=None):
        """Store a transition for all agents."""
        self.buffer.add(observations, actions, rewards, next_observations, dones,
                        prev_joint_action, prev_observations)

    def update(self, batch_size, gamma):
        """
        Perform one update step for all agents.

        Args:
            batch_size: Number of transitions to sample
            gamma: Discount factor
        """
        batch = self.buffer.sample(batch_size)

        # Compute target actions for all agents using target networks
        # For prev_obs: at next_state, the "previous obs" is the current obs
        target_actions = {}
        for agent_id in self.agent_ids:
            if self.use_prev_obs:
                # Concatenate current obs (as prev) with next obs for target actor
                target_actor_input = torch.cat([batch['obs'][agent_id], batch['next_obs'][agent_id]], dim=1)
            else:
                target_actor_input = batch['next_obs'][agent_id]
            target_actions[agent_id] = self.agents[agent_id].get_target_action(target_actor_input)

        # Zero shared actor gradients before updates
        if self.shared_actor:
            self.shared_agent_actor_optimizer.zero_grad()
            if self.shared_adversary_actor_optimizer is not None:
                self.shared_adversary_actor_optimizer.zero_grad()

        for agent_id in self.agent_ids:
            self._update_agent(agent_id, batch, target_actions, gamma)

        # Step shared actor optimizers after all agent updates
        if self.shared_actor:
            torch.nn.utils.clip_grad_norm_(self.shared_agent_actor.parameters(), 0.5)
            self.shared_agent_actor_optimizer.step()
            if self.shared_adversary_actor_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.shared_adversary_actor.parameters(), 0.5)
                self.shared_adversary_actor_optimizer.step()

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
        # Concatenate prev_obs with current obs if use_prev_obs is enabled
        if self.use_prev_obs:
            actor_obs = torch.cat([batch['prev_obs'][agent_id], batch['obs'][agent_id]], dim=1)
        else:
            actor_obs = batch['obs'][agent_id]

        action, current_logits = self.agents[agent_id].actor.get_action(
            actor_obs, explore=True, return_logits=True
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

        # For shared actor: just accumulate gradients, optimizer step happens in update()
        if self.shared_actor:
            total_actor_loss.backward()
        else:
            agent.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
            agent.actor_optimizer.step()

    def soft_update_targets(self, tau):
        """Soft update all target networks."""
        if self.shared_actor:
            # Update shared target actors once (not per-agent)
            self._soft_update(self.shared_agent_target_actor, self.shared_agent_actor, tau)
            if self.shared_adversary_actor is not None:
                self._soft_update(self.shared_adversary_target_actor, self.shared_adversary_actor, tau)
            # Update individual critics
            for agent in self.agents.values():
                self._soft_update(agent.target_critic, agent.critic, tau)
        else:
            # Original behavior: update both actor and critic per agent
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
        if self.shared_actor:
            state = {
                'shared_actor': True,
                'shared_agent_actor': self.shared_agent_actor.state_dict(),
                'shared_adversary_actor': self.shared_adversary_actor.state_dict() if self.shared_adversary_actor else None,
                'agent_team': self.agent_team,
                'adversary_team': self.adversary_team,
                'critics': {
                    agent_id: agent.critic.state_dict()
                    for agent_id, agent in self.agents.items()
                }
            }
        else:
            state = {
                agent_id: {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict()
                }
                for agent_id, agent in self.agents.items()
            }
            state['shared_actor'] = False
        torch.save(state, os.path.join(path, 'model.pt'))

    def load(self, path):
        """Load model weights."""
        state = torch.load(os.path.join(path, 'model.pt'), map_location=self.device)

        # Handle both shared_actor and individual actor models
        is_shared = state.get('shared_actor', False)

        if is_shared and self.shared_actor:
            # Load shared actor weights
            self.shared_agent_actor.load_state_dict(state['shared_agent_actor'])
            self.shared_agent_target_actor = deepcopy(self.shared_agent_actor)

            if self.shared_adversary_actor is not None and state['shared_adversary_actor'] is not None:
                self.shared_adversary_actor.load_state_dict(state['shared_adversary_actor'])
                self.shared_adversary_target_actor = deepcopy(self.shared_adversary_actor)

            # Load individual critic weights
            for agent_id, agent in self.agents.items():
                agent.critic.load_state_dict(state['critics'][agent_id])
                agent.target_critic = deepcopy(agent.critic)
        else:
            # Load individual actor/critic weights (original format)
            for agent_id, agent in self.agents.items():
                agent.actor.load_state_dict(state[agent_id]['actor'])
                agent.critic.load_state_dict(state[agent_id]['critic'])
                agent.target_actor = deepcopy(agent.actor)
                agent.target_critic = deepcopy(agent.critic)
