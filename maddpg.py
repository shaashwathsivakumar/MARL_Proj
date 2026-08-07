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
                 shared_actor=None, shared_target_actor=None, use_twin_critic=False):
        self.device = device
        self.shared_actor_mode = shared_actor is not None
        self.use_twin_critic = use_twin_critic

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

        # TD3-style twin critic: a second, independently-initialized critic
        # used (via min of both targets) to reduce Q-value overestimation bias.
        if use_twin_critic:
            self.critic2 = Critic(critic_input_dim).to(device)
            self.target_critic2 = deepcopy(self.critic2)
            self.critic2_optimizer = Adam(self.critic2.parameters(), lr=critic_lr)

    def select_action(self, actor_input, explore=True):
        """
        Select action given pre-built actor input (obs, or obs+a_hat, or prev_obs+obs).

        Args:
            actor_input: Numpy array — already assembled by MADDPG.select_actions
            explore: Whether to use stochastic action selection

        Returns:
            Integer action index
        """
        obs_tensor = torch.from_numpy(
            np.array(actor_input, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
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
                 use_prev_obs=False, shared_actor=False, env_name=None,
                 ai_net=None, sub_assignments=None, online_ai_net=False,
                 use_adversary_gating=False, gating_temperature=0.5,
                 gating_ema_decay=0.05, use_twin_critic=False, policy_delay=2):
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
            env_name: Environment name (required if shared_actor=True or
                use_adversary_gating=True, for team detection)
            ai_net: Optional pre-trained AINet instance (Algorithm 3)
            sub_assignments: Dict mapping agent_index -> sub_assign tensor (required if ai_net)
            use_adversary_gating: Experimental — scale each opponent agent's obs/action
                features in the critic input by a "trust" gate derived from how much that
                opponent's action distribution has drifted from its own slow-moving EMA.
                Intuition: the centralized critic's model of the opponent is least
                reliable right when the opponent's policy is changing fastest, so its
                contribution is downweighted then. No effect on agents with no
                opponent team (e.g. fully cooperative envs).
            gating_temperature: Controls sensitivity of the gate to drift; smaller values
                attenuate more aggressively for the same drift magnitude.
            gating_ema_decay: Update rate of the slow-moving per-agent action EMA used
                as the drift reference.
            use_twin_critic: TD3-style (Fujimoto et al. 2018) clipped double Q-learning —
                each agent gets two independently-initialized critics; the TD target uses
                the minimum of both targets' predictions (reduces Q-value overestimation
                bias), and actor + target-network updates are delayed to every
                `policy_delay` calls to update() (matches the published algorithm). Actor
                loss still uses critic 1 only. Target policy smoothing (the third TD3
                component) is intentionally omitted — it's designed for continuous action
                noise and has no clean analog for this codebase's discrete Gumbel-Softmax
                actions.
            policy_delay: Number of update() calls between actor/target-network updates
                when use_twin_critic is True. Ignored otherwise (every call updates).
        """
        self.agent_ids = agent_ids
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.device = device
        self.use_prev_action = use_prev_action
        self.use_adversary_gating = use_adversary_gating
        self.gating_temperature = gating_temperature
        self.gating_ema_decay = gating_ema_decay
        self.use_twin_critic = use_twin_critic
        self.policy_delay = policy_delay
        self._update_step_count = 0
        self._should_soft_update = True
        # AI_Net requires prev_obs to be stored in the buffer
        self.ai_net = ai_net
        self.sub_assignments = sub_assignments
        self.use_ai_net    = ai_net is not None
        self.online_ai_net = online_ai_net and (ai_net is not None)
        if self.use_ai_net:
            use_prev_obs = True
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

        # Determine team membership (needed for shared_actor and/or adversary_gating)
        if self.shared_actor or self.use_adversary_gating:
            self.agent_team = get_agent_team(env_name, agent_ids) if env_name else agent_ids
            self.adversary_team = [a for a in agent_ids if a not in self.agent_team]
        else:
            self.agent_team = None
            self.adversary_team = None

        # Precompute, for each agent, the set of agent_ids on the *other* team
        # (used by adversary gating; empty/unused when gating is disabled or the
        # env has no adversary team, e.g. fully cooperative envs).
        if self.use_adversary_gating and self.adversary_team:
            agent_team_set = set(self.agent_team)
            adversary_team_set = set(self.adversary_team)
            self._opponent_ids = {
                aid: (adversary_team_set if aid in agent_team_set else agent_team_set)
                for aid in agent_ids
            }
        else:
            self._opponent_ids = {}

        # Per-agent offsets into the flat prev_joint_action tensor (matches the
        # concatenation order used when prev_joint_action is built in train.py).
        self._action_offsets = {}
        _offset = 0
        for aid in agent_ids:
            self._action_offsets[aid] = _offset
            _offset += action_dims[aid]

        # Slow-moving EMA of each agent's batch-mean action distribution, used as
        # the drift reference for adversary gating. Lazily initialized on first use.
        self.action_emas = {aid: None for aid in agent_ids}
        self._current_gates = {}

        # Setup shared actors if enabled
        if self.shared_actor:
            # Create shared actor for agent team
            if self.use_ai_net:
                agent_type = ai_net.agent_types[agent_ids.index(self.agent_team[0])]
                agent_obs_dim = obs_dims[self.agent_team[0]] + ai_net.total_action_dim_by_type[agent_type]
            elif use_prev_obs:
                agent_obs_dim = obs_dims[self.agent_team[0]] * 2
            else:
                agent_obs_dim = obs_dims[self.agent_team[0]]
            agent_action_dim = action_dims[self.agent_team[0]]
            self.shared_agent_actor = Actor(agent_obs_dim, agent_action_dim).to(device)
            self.shared_agent_target_actor = deepcopy(self.shared_agent_actor)
            self.shared_agent_actor_optimizer = Adam(self.shared_agent_actor.parameters(), lr=actor_lr)

            # Create shared actor for adversary team (if any)
            if self.adversary_team:
                if self.use_ai_net:
                    adv_type = ai_net.agent_types[agent_ids.index(self.adversary_team[0])]
                    adv_obs_dim = obs_dims[self.adversary_team[0]] + ai_net.total_action_dim_by_type[adv_type]
                elif use_prev_obs:
                    adv_obs_dim = obs_dims[self.adversary_team[0]] * 2
                else:
                    adv_obs_dim = obs_dims[self.adversary_team[0]]
                adv_action_dim = action_dims[self.adversary_team[0]]
                self.shared_adversary_actor = Actor(adv_obs_dim, adv_action_dim).to(device)
                self.shared_adversary_target_actor = deepcopy(self.shared_adversary_actor)
                self.shared_adversary_actor_optimizer = Adam(self.shared_adversary_actor.parameters(), lr=actor_lr)
            else:
                self.shared_adversary_actor = None
                self.shared_adversary_target_actor = None
                self.shared_adversary_actor_optimizer = None

        # Create agents
        self.agents = {}
        for agent_id in agent_ids:
            if self.use_ai_net:
                agent_type_i = ai_net.agent_types[agent_ids.index(agent_id)]
                actor_obs_dim = obs_dims[agent_id] + ai_net.total_action_dim_by_type[agent_type_i]
            elif use_prev_obs:
                actor_obs_dim = obs_dims[agent_id] * 2
            else:
                actor_obs_dim = obs_dims[agent_id]

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
                    shared_target_actor=shared_target_actor_net,
                    use_twin_critic=self.use_twin_critic
                )
            else:
                # Individual actors (original behavior)
                self.agents[agent_id] = MADDPGAgent(
                    actor_obs_dim,
                    action_dims[agent_id],
                    critic_input_dim,
                    actor_lr,
                    critic_lr,
                    device,
                    use_twin_critic=self.use_twin_critic
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
        actions = {}
        for i, agent_id in enumerate(self.agent_ids):
            obs = observations[agent_id]

            if self.use_ai_net:
                if prev_observations is not None:
                    obs_t      = torch.tensor(obs, dtype=torch.float32).to(self.device)
                    prev_obs_t = torch.tensor(
                        prev_observations[agent_id], dtype=torch.float32
                    ).to(self.device)
                    sub_assign = self.sub_assignments[i].to(self.device)
                    with torch.no_grad():
                        a_hat = self.ai_net.forward(sub_assign, i, obs_t, prev_obs_t)
                    actor_input = np.concatenate([obs, a_hat.cpu().numpy()])
                else:
                    # First step of episode: no previous obs, use zero vector
                    agent_type_i = self.ai_net.agent_types[i]
                    actor_input = np.concatenate(
                        [obs, np.zeros(self.ai_net.total_action_dim_by_type[agent_type_i], dtype=np.float32)]
                    )
            elif self.use_prev_obs and prev_observations is not None:
                actor_input = np.concatenate([prev_observations[agent_id], obs])
            else:
                actor_input = obs

            actions[agent_id] = self.agents[agent_id].select_action(actor_input, explore)
        return actions

    def store_transition(self, observations, actions, rewards, next_observations, dones,
                         prev_joint_action=None, prev_observations=None):
        """Store a transition for all agents."""
        self.buffer.add(observations, actions, rewards, next_observations, dones,
                        prev_joint_action, prev_observations)

    def _update_ai_net_online(self, batch):
        """
        Update AINet on the current replay buffer batch (simultaneous training).

        Uses the same transitions just sampled for the MADDPG update so that
        AINet always tracks the current policy's behavior.

        For each (observer, observed) pair:
          curr_obs  = batch['next_obs'][observer]   (observation at t+1)
          last_obs  = batch['obs'][observer]         (observation at t)
          target    = batch['actions'][observed]     (one-hot action taken at t,
                      which caused the obs_t -> next_obs_t transition)
        """
        for observer_ind, observer_id in enumerate(self.agent_ids):
            observer_type = self.ai_net.agent_types[observer_ind]
            sub_assign    = self.sub_assignments[observer_ind].to(self.device)

            curr_obs_batch = batch['next_obs'][observer_id]
            last_obs_batch = batch['obs'][observer_id]

            slfgbl_mask = sub_assign == -1
            curr_slfgbl = curr_obs_batch[:, slfgbl_mask]
            last_slfgbl = last_obs_batch[:, slfgbl_mask]

            for observed_ind, observed_id in enumerate(self.agent_ids):
                observed_type = self.ai_net.agent_types[observed_ind]
                target = batch['actions'][observed_id]  # (B, action_dim), already one-hot

                if observed_ind == observer_ind:
                    bundle = torch.cat([curr_slfgbl, last_slfgbl,
                                        curr_slfgbl - last_slfgbl], dim=1)
                    a_pred = self.ai_net.self_modules[observer_type](bundle)
                    loss   = F.mse_loss(a_pred, target)
                    self.ai_net.self_modules[observer_type].update(loss)
                else:
                    if observed_type == observer_type and self.ai_net.fellow_mode != 'action':
                        # Fellow-same-type slots predict displacement, not action,
                        # in velocity modes -- not supported for online updates.
                        continue

                    other_mask  = sub_assign == observed_ind
                    curr_other  = curr_obs_batch[:, other_mask]
                    last_other  = last_obs_batch[:, other_mask]
                    curr_bundle = torch.cat([curr_other, curr_slfgbl], dim=1)
                    last_bundle = torch.cat([last_other, last_slfgbl], dim=1)
                    bundle      = torch.cat([curr_bundle, last_bundle,
                                             curr_bundle - last_bundle], dim=1)
                    a_pred = self.ai_net.social_modules[observer_type][observed_type](bundle)
                    loss   = F.mse_loss(a_pred, target)
                    self.ai_net.social_modules[observer_type][observed_type].update(loss)

    def _compute_gates(self, batch):
        """
        Compute, for each agent, a trust gate in (0, 1] based on how far this
        update's batch-mean action distribution has drifted from that agent's
        own slow-moving EMA of past batch-mean actions.

        Gate is 1.0 (no attenuation) on the very first call for each agent
        (no drift reference yet) and whenever drift is zero. The EMA is
        updated once per call, using the same drift measurement.
        """
        gates = {}
        for aid in self.agent_ids:
            batch_mean_action = batch['actions'][aid].mean(dim=0).detach()
            ema = self.action_emas[aid]
            if ema is None:
                gates[aid] = 1.0
            else:
                drift = torch.norm(batch_mean_action - ema, p=2).item()
                gates[aid] = float(np.exp(-drift / self.gating_temperature))
            if ema is None:
                self.action_emas[aid] = batch_mean_action.clone()
            else:
                self.action_emas[aid] = (
                    (1 - self.gating_ema_decay) * ema
                    + self.gating_ema_decay * batch_mean_action
                )
        return gates

    def _gate_for(self, owner_id, other_id):
        """Trust gate applied to other_id's obs/action features within owner_id's
        critic input. 1.0 (no-op) unless adversary gating is on and other_id is
        on the opposing team from owner_id."""
        if not self.use_adversary_gating:
            return 1.0
        if other_id not in self._opponent_ids.get(owner_id, ()):
            return 1.0
        return self._current_gates.get(other_id, 1.0)

    def update(self, batch_size, gamma):
        """
        Perform one update step for all agents.

        Args:
            batch_size: Number of transitions to sample
            gamma: Discount factor
        """
        batch = self.buffer.sample(batch_size)

        if self.use_adversary_gating:
            self._current_gates = self._compute_gates(batch)

        # TD3-style delayed policy updates: actor and target-network updates only
        # happen every `policy_delay` calls to update(); critics update every call.
        if self.use_twin_critic:
            self._update_step_count += 1
            update_actor = (self._update_step_count % self.policy_delay == 0)
        else:
            update_actor = True
        self._should_soft_update = update_actor

        # Compute target actions for all agents using target networks
        # AI_Net: â^{j+1} = AI_Net(next_obs, curr_obs)  → actor sees [next_obs, â^{j+1}]
        # prev_obs mode: actor sees [curr_obs (as prev), next_obs]
        target_actions = {}
        for i, agent_id in enumerate(self.agent_ids):
            if self.use_ai_net:
                sub_assign = self.sub_assignments[i].to(self.device)
                with torch.no_grad():
                    a_hat_next = self.ai_net.forward_batch(
                        sub_assign, i,
                        batch['next_obs'][agent_id],
                        batch['obs'][agent_id]   # current obs is "prev" for next step
                    )
                target_actor_input = torch.cat(
                    [batch['next_obs'][agent_id], a_hat_next], dim=1
                )
            elif self.use_prev_obs:
                target_actor_input = torch.cat(
                    [batch['obs'][agent_id], batch['next_obs'][agent_id]], dim=1
                )
            else:
                target_actor_input = batch['next_obs'][agent_id]
            target_actions[agent_id] = self.agents[agent_id].get_target_action(target_actor_input)

        # Zero shared actor gradients before updates
        if self.shared_actor and update_actor:
            self.shared_agent_actor_optimizer.zero_grad()
            if self.shared_adversary_actor_optimizer is not None:
                self.shared_adversary_actor_optimizer.zero_grad()

        for agent_id in self.agent_ids:
            self._update_agent(agent_id, batch, target_actions, gamma, update_actor)

        # Step shared actor optimizers after all agent updates
        if self.shared_actor and update_actor:
            torch.nn.utils.clip_grad_norm_(self.shared_agent_actor.parameters(), 0.5)
            self.shared_agent_actor_optimizer.step()
            if self.shared_adversary_actor_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.shared_adversary_actor.parameters(), 0.5)
                self.shared_adversary_actor_optimizer.step()

        if self.online_ai_net:
            self._update_ai_net_online(batch)

    def _update_agent(self, agent_id, batch, target_actions, gamma, update_actor=True):
        """Update a single agent's critic (every call) and actor (only when
        update_actor is True — always True unless use_twin_critic delays it)."""
        agent = self.agents[agent_id]

        # === Critic Update ===
        # Current Q-value: Q(s, a) where s and a are all agents' obs/actions.
        # Adversary gating (if enabled): opponent agents' obs/action features are
        # scaled by a per-opponent trust gate before concatenation; own-team
        # features (including this agent's own) are always passed through
        # unscaled (gate == 1.0).
        current_obs = torch.cat(
            [batch['obs'][aid] * self._gate_for(agent_id, aid) for aid in self.agent_ids], dim=1
        )
        current_actions = torch.cat(
            [batch['actions'][aid] * self._gate_for(agent_id, aid) for aid in self.agent_ids], dim=1
        )

        # Build critic input (optionally include previous joint action)
        if self.use_prev_action:
            prev_joint_action = batch['prev_joint_action']
            if self.use_adversary_gating:
                prev_joint_action = torch.cat([
                    prev_joint_action[:, self._action_offsets[aid]:
                                       self._action_offsets[aid] + self.action_dims[aid]]
                    * self._gate_for(agent_id, aid)
                    for aid in self.agent_ids
                ], dim=1)
            critic_input = torch.cat([current_obs, prev_joint_action, current_actions], dim=1)
        else:
            critic_input = torch.cat([current_obs, current_actions], dim=1)

        current_q = agent.critic(critic_input).squeeze(1)
        if self.use_twin_critic:
            current_q2 = agent.critic2(critic_input).squeeze(1)

        # Target Q-value: r + gamma * Q'(s', a') * (1 - done)
        next_obs = torch.cat(
            [batch['next_obs'][aid] * self._gate_for(agent_id, aid) for aid in self.agent_ids], dim=1
        )
        next_actions = torch.cat(
            [target_actions[aid] * self._gate_for(agent_id, aid) for aid in self.agent_ids], dim=1
        )

        with torch.no_grad():
            # For target: prev_action at next state = current_actions
            if self.use_prev_action:
                target_critic_input = torch.cat([next_obs, current_actions, next_actions], dim=1)
            else:
                target_critic_input = torch.cat([next_obs, next_actions], dim=1)

            if self.use_twin_critic:
                # Clipped double Q-learning (TD3): use the min of both target
                # critics to counter Q-value overestimation bias.
                target_q1 = agent.target_critic(target_critic_input).squeeze(1)
                target_q2 = agent.target_critic2(target_critic_input).squeeze(1)
                target_q = torch.min(target_q1, target_q2)
            else:
                target_q = agent.target_critic(target_critic_input).squeeze(1)
            td_target = batch['rewards'][agent_id] + gamma * target_q * (1 - batch['dones'][agent_id])

        if self.use_twin_critic:
            critic_loss = F.mse_loss(current_q, td_target) + F.mse_loss(current_q2, td_target)
            agent.critic_optimizer.zero_grad()
            agent.critic2_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(agent.critic2.parameters(), 0.5)
            agent.critic_optimizer.step()
            agent.critic2_optimizer.step()
        else:
            critic_loss = F.mse_loss(current_q, td_target)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
            agent.critic_optimizer.step()

        if not update_actor:
            return

        # === Actor Update ===
        # Build actor input for current agent:
        #   AI_Net mode:    [obs, â]  where â = AI_Net(obs, prev_obs)
        #   prev_obs mode:  [prev_obs, obs]
        #   standard:       obs
        agent_idx = self.agent_ids.index(agent_id)
        if self.use_ai_net:
            sub_assign = self.sub_assignments[agent_idx].to(self.device)
            with torch.no_grad():
                a_hat = self.ai_net.forward_batch(
                    sub_assign, agent_idx,
                    batch['obs'][agent_id],
                    batch['prev_obs'][agent_id]
                )
            actor_obs = torch.cat([batch['obs'][agent_id], a_hat], dim=1)
        elif self.use_prev_obs:
            actor_obs = torch.cat([batch['prev_obs'][agent_id], batch['obs'][agent_id]], dim=1)
        else:
            actor_obs = batch['obs'][agent_id]

        action, current_logits = self.agents[agent_id].actor.get_action(
            actor_obs, explore=True, return_logits=True
        )

        # Build action list: current agent uses fresh action, others use buffer
        # actions (gated by opponent trust, same as the critic update above).
        actor_actions = []
        for aid in self.agent_ids:
            if aid == agent_id:
                actor_actions.append(action)
            else:
                actor_actions.append(batch['actions'][aid] * self._gate_for(agent_id, aid))

        # Actor loss: maximize Q-value (critic 1 only, matching TD3 convention)
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
        """Soft update all target networks.

        With use_twin_critic, this is a no-op except on the same delayed
        schedule as the actor update (matches TD3: targets only move when
        the policy does)."""
        if self.use_twin_critic and not self._should_soft_update:
            return

        if self.shared_actor:
            # Update shared target actors once (not per-agent)
            self._soft_update(self.shared_agent_target_actor, self.shared_agent_actor, tau)
            if self.shared_adversary_actor is not None:
                self._soft_update(self.shared_adversary_target_actor, self.shared_adversary_actor, tau)
            # Update individual critics
            for agent in self.agents.values():
                self._soft_update(agent.target_critic, agent.critic, tau)
                if self.use_twin_critic:
                    self._soft_update(agent.target_critic2, agent.critic2, tau)
        else:
            # Original behavior: update both actor and critic per agent
            for agent in self.agents.values():
                self._soft_update(agent.target_actor, agent.actor, tau)
                self._soft_update(agent.target_critic, agent.critic, tau)
                if self.use_twin_critic:
                    self._soft_update(agent.target_critic2, agent.critic2, tau)

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
            if self.use_twin_critic:
                state['critics2'] = {
                    agent_id: agent.critic2.state_dict()
                    for agent_id, agent in self.agents.items()
                }
        else:
            state = {
                agent_id: {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    **({'critic2': agent.critic2.state_dict()} if self.use_twin_critic else {})
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
                if self.use_twin_critic and 'critics2' in state:
                    agent.critic2.load_state_dict(state['critics2'][agent_id])
                    agent.target_critic2 = deepcopy(agent.critic2)
        else:
            # Load individual actor/critic weights (original format)
            for agent_id, agent in self.agents.items():
                agent.actor.load_state_dict(state[agent_id]['actor'])
                agent.critic.load_state_dict(state[agent_id]['critic'])
                agent.target_actor = deepcopy(agent.actor)
                agent.target_critic = deepcopy(agent.critic)
                if self.use_twin_critic and 'critic2' in state[agent_id]:
                    agent.critic2.load_state_dict(state[agent_id]['critic2'])
                    agent.target_critic2 = deepcopy(agent.critic2)
