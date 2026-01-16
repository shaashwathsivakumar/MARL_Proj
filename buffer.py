"""
Replay buffer for MADDPG training.
"""
import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-size replay buffer for a single agent.
    Stores transitions and provides batch sampling.
    """
    def __init__(self, capacity, obs_dim, action_dim, device='cpu'):
        self.capacity = capacity
        self.device = device

        # Pre-allocate memory for efficiency
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._ptr = 0  # Current write position
        self._size = 0  # Current buffer size

    def add(self, obs, action, reward, next_obs, done):
        """Add a transition to the buffer."""
        self.observations[self._ptr] = obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.next_observations[self._ptr] = next_obs
        self.dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_indices(self, batch_size):
        """Generate random indices for sampling."""
        return np.random.choice(self._size, size=batch_size, replace=False)

    def get_batch(self, indices):
        """
        Retrieve a batch of transitions at given indices.

        Returns:
            Tuple of tensors: (obs, actions, rewards, next_obs, dones)
        """
        obs = torch.from_numpy(self.observations[indices]).to(self.device)
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_obs = torch.from_numpy(self.next_observations[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return self._size


class MultiAgentReplayBuffer:
    """
    Collection of replay buffers, one per agent.
    Ensures synchronized sampling across all agents.
    """
    def __init__(self, agent_ids, capacity, obs_dims, action_dims, device='cpu'):
        """
        Args:
            agent_ids: List of agent identifiers
            capacity: Maximum buffer size
            obs_dims: Dict mapping agent_id -> observation dimension
            action_dims: Dict mapping agent_id -> action dimension
            device: Torch device for tensor conversion
        """
        self.agent_ids = agent_ids
        self.buffers = {
            agent_id: ReplayBuffer(capacity, obs_dims[agent_id], action_dims[agent_id], device)
            for agent_id in agent_ids
        }

    def add(self, observations, actions, rewards, next_observations, dones):
        """
        Add transitions for all agents.

        Args:
            observations: Dict of observations per agent
            actions: Dict of actions per agent (as one-hot or will be converted)
            rewards: Dict of rewards per agent
            next_observations: Dict of next observations per agent
            dones: Dict of done flags per agent
        """
        for agent_id in self.agent_ids:
            action = actions[agent_id]
            # Convert integer action to one-hot if needed
            if isinstance(action, (int, np.integer)):
                action_dim = self.buffers[agent_id].actions.shape[1]
                action_onehot = np.zeros(action_dim, dtype=np.float32)
                action_onehot[action] = 1.0
                action = action_onehot

            self.buffers[agent_id].add(
                observations[agent_id],
                action,
                rewards[agent_id],
                next_observations[agent_id],
                dones[agent_id]
            )

    def sample(self, batch_size):
        """
        Sample synchronized batches from all agent buffers.

        Returns:
            Dict with keys: 'obs', 'actions', 'rewards', 'next_obs', 'dones'
            Each value is a dict mapping agent_id -> tensor
        """
        # Use same indices for all agents to keep transitions aligned
        indices = self.buffers[self.agent_ids[0]].sample_indices(batch_size)

        batch = {
            'obs': {},
            'actions': {},
            'rewards': {},
            'next_obs': {},
            'dones': {}
        }

        for agent_id in self.agent_ids:
            obs, actions, rewards, next_obs, dones = self.buffers[agent_id].get_batch(indices)
            batch['obs'][agent_id] = obs
            batch['actions'][agent_id] = actions
            batch['rewards'][agent_id] = rewards
            batch['next_obs'][agent_id] = next_obs
            batch['dones'][agent_id] = dones

        return batch

    def __len__(self):
        return len(self.buffers[self.agent_ids[0]])
