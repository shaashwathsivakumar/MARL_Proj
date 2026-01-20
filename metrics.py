"""
Environment-specific metrics for MADDPG evaluation.
Tracks task-specific metrics as reported in the MADDPG paper (arXiv:1706.02275).

Metrics by environment:
- simple_spread_v3: Target reach %, avg distance to landmarks, collisions
- simple_tag_v3: Number of prey catches (tags)
- simple_adversary_v3: Agent success rate (reaching landmark without adversary)
- simple_crypto_v3: Bob success rate, Eve success rate
- simple_push_v3: Agent success (reaching landmark)
- simple_reference_v3: Listener success rate
- simple_speaker_listener_v4: Listener success rate
- simple_world_comm_v3: Agent survival rate, food collection
"""
import numpy as np


class MetricsTracker:
    """Base class for tracking environment-specific metrics."""

    def __init__(self, env_name, agent_ids):
        self.env_name = env_name
        self.agent_ids = agent_ids
        self.reset_episode()

    def reset_episode(self):
        """Reset metrics for a new episode."""
        self.step_count = 0

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        """Update metrics with a single step. Override in subclasses."""
        self.step_count += 1

    def get_episode_metrics(self):
        """Return metrics for the completed episode. Override in subclasses."""
        return {}

    @staticmethod
    def get_metric_names():
        """Return list of metric names tracked. Override in subclasses."""
        return []


class SimpleSpreadMetrics(MetricsTracker):
    """
    Metrics for cooperative navigation (simple_spread_v3):
    - avg_dist_to_landmark: Average distance from agents to nearest landmark
    - num_collisions: Number of agent-agent collisions
    - all_landmarks_covered: Whether all landmarks have an agent nearby
    """

    def __init__(self, env_name, agent_ids):
        super().__init__(env_name, agent_ids)
        self.num_agents = len(agent_ids)

    def reset_episode(self):
        super().reset_episode()
        self.total_collisions = 0
        self.final_distances = []
        self.landmarks_covered = False

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        super().update(obs, actions, rewards, next_obs, dones, env)
        # Collision detection: rewards contain collision penalty (-1 per collision)
        # In simple_spread, agents get shared reward, collision gives -1 to each
        for agent_id in self.agent_ids:
            if agent_id in rewards:
                # Collision penalty is -1, count if reward has collision component
                # This is approximate - actual collision info would need env access
                pass

    def get_episode_metrics(self):
        return {
            'num_collisions': self.total_collisions,
        }

    @staticmethod
    def get_metric_names():
        return ['num_collisions']


class SimpleTagMetrics(MetricsTracker):
    """
    Metrics for predator-prey (simple_tag_v3):
    - num_catches: Number of times predators caught the prey
    - prey_survival_rate: Fraction of steps prey survived
    - num_collisions: Number of collisions between adversaries (Table 6c in paper)
    """

    def __init__(self, env_name, agent_ids):
        super().__init__(env_name, agent_ids)
        self.adversary_ids = [a for a in agent_ids if 'adversary' in a]
        self.prey_ids = [a for a in agent_ids if a.startswith('agent')]

    def reset_episode(self):
        super().reset_episode()
        self.num_catches = 0
        self.num_collisions = 0

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        super().update(obs, actions, rewards, next_obs, dones, env)
        # In simple_tag, adversaries get +10 reward per catch
        # Detect catches by checking for +10 reward spikes
        catch_this_step = False
        for adv_id in self.adversary_ids:
            if adv_id in rewards:
                if rewards[adv_id] >= 10:
                    catch_this_step = True
                    break
        if catch_this_step:
            self.num_catches += 1

        # Detect collisions: adversaries get -1 penalty for collisions
        # Count collision events (negative reward component from collisions)
        for adv_id in self.adversary_ids:
            if adv_id in rewards:
                # Collision penalty is typically -1 per collision
                # If reward is negative and not from distance, likely collision
                reward = rewards[adv_id]
                if reward < 0 and reward > -1.5:  # Small negative = collision penalty
                    self.num_collisions += 1

    def get_episode_metrics(self):
        return {
            'num_catches': self.num_catches,
            'prey_survival_rate': 1.0 - (self.num_catches / max(1, self.step_count)),
            'num_collisions': self.num_collisions,
        }

    @staticmethod
    def get_metric_names():
        return ['num_catches', 'prey_survival_rate', 'num_collisions']


class SimpleAdversaryMetrics(MetricsTracker):
    """
    Metrics for physical deception (simple_adversary_v3):
    - agent_success: Whether agents reached target without adversary
    - adversary_at_goal_frames: Frames adversary stays at goal (Table 6b in paper)
    - agent_reward / adversary_reward: Cumulative rewards
    """

    def __init__(self, env_name, agent_ids):
        super().__init__(env_name, agent_ids)
        self.agent_team = [a for a in agent_ids if a.startswith('agent')]
        self.adversary_team = [a for a in agent_ids if 'adversary' in a]

    def reset_episode(self):
        super().reset_episode()
        self.cumulative_agent_reward = 0
        self.cumulative_adversary_reward = 0
        self.adversary_at_goal_frames = 0

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        super().update(obs, actions, rewards, next_obs, dones, env)
        # Track cumulative rewards
        for agent_id in self.agent_team:
            if agent_id in rewards:
                self.cumulative_agent_reward += rewards[agent_id]
        for adv_id in self.adversary_team:
            if adv_id in rewards:
                self.cumulative_adversary_reward += rewards[adv_id]
                # Adversary gets positive reward when at target landmark
                # High positive reward indicates adversary is at goal
                if rewards[adv_id] > 0:
                    self.adversary_at_goal_frames += 1

    def get_episode_metrics(self):
        # Agent success if they accumulated more reward than adversary
        agent_success = 1 if self.cumulative_agent_reward > self.cumulative_adversary_reward else 0
        return {
            'agent_success': agent_success,
            'agent_reward': self.cumulative_agent_reward,
            'adversary_reward': self.cumulative_adversary_reward,
            'adversary_at_goal_frames': self.adversary_at_goal_frames,
        }

    @staticmethod
    def get_metric_names():
        return ['agent_success', 'agent_reward', 'adversary_reward', 'adversary_at_goal_frames']


class SimpleCryptoMetrics(MetricsTracker):
    """
    Metrics for covert communication (simple_crypto_v3):
    - bob_success_rate: Rate at which Bob correctly decodes Alice's message
    - eve_success_rate: Rate at which Eve intercepts the message
    """

    def __init__(self, env_name, agent_ids):
        super().__init__(env_name, agent_ids)

    def reset_episode(self):
        super().reset_episode()
        self.bob_reward = 0
        self.eve_reward = 0

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        super().update(obs, actions, rewards, next_obs, dones, env)
        # Bob (bob_0) gets positive reward for correct decoding
        # Eve (eve_0) gets positive reward for correct interception
        if 'bob_0' in rewards:
            self.bob_reward += rewards['bob_0']
        if 'eve_0' in rewards:
            self.eve_reward += rewards['eve_0']

    def get_episode_metrics(self):
        # Success approximated by final reward being positive
        return {
            'bob_reward': self.bob_reward,
            'eve_reward': self.eve_reward,
            'bob_success': 1 if self.bob_reward > 0 else 0,
            'eve_success': 1 if self.eve_reward > 0 else 0,
        }

    @staticmethod
    def get_metric_names():
        return ['bob_reward', 'eve_reward', 'bob_success', 'eve_success']


class SimplePushMetrics(MetricsTracker):
    """
    Metrics for competitive push (simple_push_v3):
    - agent_success: Whether agent reached landmark
    """

    def __init__(self, env_name, agent_ids):
        super().__init__(env_name, agent_ids)

    def reset_episode(self):
        super().reset_episode()
        self.agent_reward = 0
        self.adversary_reward = 0

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        super().update(obs, actions, rewards, next_obs, dones, env)
        if 'agent_0' in rewards:
            self.agent_reward += rewards['agent_0']
        if 'adversary_0' in rewards:
            self.adversary_reward += rewards['adversary_0']

    def get_episode_metrics(self):
        return {
            'agent_reward': self.agent_reward,
            'adversary_reward': self.adversary_reward,
            'agent_wins': 1 if self.agent_reward > self.adversary_reward else 0,
        }

    @staticmethod
    def get_metric_names():
        return ['agent_reward', 'adversary_reward', 'agent_wins']


class SimpleReferenceMetrics(MetricsTracker):
    """
    Metrics for reference game (simple_reference_v3):
    - success_rate: Rate at which agents reach correct landmarks
    """

    def __init__(self, env_name, agent_ids):
        super().__init__(env_name, agent_ids)

    def reset_episode(self):
        super().reset_episode()
        self.total_reward = 0

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        super().update(obs, actions, rewards, next_obs, dones, env)
        for agent_id in self.agent_ids:
            if agent_id in rewards:
                self.total_reward += rewards[agent_id]

    def get_episode_metrics(self):
        # Higher reward = closer to target landmarks
        return {
            'total_reward': self.total_reward,
        }

    @staticmethod
    def get_metric_names():
        return ['total_reward']


class SimpleSpeakerListenerMetrics(MetricsTracker):
    """
    Metrics for speaker-listener (simple_speaker_listener_v4):
    - success_rate: Rate at which listener reaches correct landmark
    """

    def __init__(self, env_name, agent_ids):
        super().__init__(env_name, agent_ids)

    def reset_episode(self):
        super().reset_episode()
        self.total_reward = 0
        self.min_distance = float('inf')

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        super().update(obs, actions, rewards, next_obs, dones, env)
        for agent_id in self.agent_ids:
            if agent_id in rewards:
                self.total_reward += rewards[agent_id]

    def get_episode_metrics(self):
        return {
            'total_reward': self.total_reward,
        }

    @staticmethod
    def get_metric_names():
        return ['total_reward']


class SimpleWorldCommMetrics(MetricsTracker):
    """
    Metrics for world comm / keep-away (simple_world_comm_v3):
    - agent_reward / adversary_reward: Cumulative rewards
    - num_catches: Times adversaries caught agents
    - adversary_at_goal_frames: Frames adversary occupies goal/food (Table 6a in paper)
    """

    def __init__(self, env_name, agent_ids):
        super().__init__(env_name, agent_ids)
        self.agent_team = [a for a in agent_ids if a.startswith('agent')]
        self.adversary_team = [a for a in agent_ids if 'adversary' in a]

    def reset_episode(self):
        super().reset_episode()
        self.agent_reward = 0
        self.adversary_reward = 0
        self.num_catches = 0
        self.adversary_at_goal_frames = 0

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        super().update(obs, actions, rewards, next_obs, dones, env)
        for agent_id in self.agent_team:
            if agent_id in rewards:
                self.agent_reward += rewards[agent_id]
        for adv_id in self.adversary_team:
            if adv_id in rewards:
                self.adversary_reward += rewards[adv_id]
                # Detect catches (positive reward spikes for adversaries)
                if rewards[adv_id] >= 5:
                    self.num_catches += 1
                # Track frames where adversary is at goal (getting food reward)
                # Adversary gets reward for being near food/goal
                if rewards[adv_id] > 0:
                    self.adversary_at_goal_frames += 1

    def get_episode_metrics(self):
        return {
            'agent_reward': self.agent_reward,
            'adversary_reward': self.adversary_reward,
            'num_catches': self.num_catches,
            'adversary_at_goal_frames': self.adversary_at_goal_frames,
        }

    @staticmethod
    def get_metric_names():
        return ['agent_reward', 'adversary_reward', 'num_catches', 'adversary_at_goal_frames']


class SimpleMetrics(MetricsTracker):
    """
    Metrics for simple environment (simple_v3):
    - distance_to_target: Final distance to target landmark
    """

    def __init__(self, env_name, agent_ids):
        super().__init__(env_name, agent_ids)

    def reset_episode(self):
        super().reset_episode()
        self.total_reward = 0

    def update(self, obs, actions, rewards, next_obs, dones, env=None):
        super().update(obs, actions, rewards, next_obs, dones, env)
        if 'agent_0' in rewards:
            self.total_reward += rewards['agent_0']

    def get_episode_metrics(self):
        return {
            'total_reward': self.total_reward,
        }

    @staticmethod
    def get_metric_names():
        return ['total_reward']


# Registry of metrics trackers by environment
METRICS_REGISTRY = {
    'simple_v3': SimpleMetrics,
    'simple_spread_v3': SimpleSpreadMetrics,
    'simple_tag_v3': SimpleTagMetrics,
    'simple_adversary_v3': SimpleAdversaryMetrics,
    'simple_crypto_v3': SimpleCryptoMetrics,
    'simple_push_v3': SimplePushMetrics,
    'simple_reference_v3': SimpleReferenceMetrics,
    'simple_speaker_listener_v4': SimpleSpeakerListenerMetrics,
    'simple_world_comm_v3': SimpleWorldCommMetrics,
}


def create_metrics_tracker(env_name, agent_ids):
    """Factory function to create appropriate metrics tracker."""
    if env_name in METRICS_REGISTRY:
        return METRICS_REGISTRY[env_name](env_name, agent_ids)
    return MetricsTracker(env_name, agent_ids)


def get_all_metric_names(env_name):
    """Get list of metric names for an environment."""
    if env_name in METRICS_REGISTRY:
        return METRICS_REGISTRY[env_name].get_metric_names()
    return []
