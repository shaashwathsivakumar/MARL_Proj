"""
Pre-Trained Action Inference (PTAI) — AI_Net implementation.

Predicts the previous joint action vector for all agents from a single
agent's current and previous observations (Section 3.2 of the paper).

Architecture:
  - Directional Social Awareness Modules: predict other agents' actions
  - Directional Self-Awareness Module:    predict the observer's own action

Pre-trained on random episodes via supervised MSE (Algorithm 2).
Frozen during MADDPG training (Algorithm 3).

Supported environments (all 9 MPE):
  simple_v3, simple_spread_v3, simple_adversary_v3, simple_crypto_v3,
  simple_push_v3, simple_reference_v3, simple_speaker_listener_v4,
  simple_tag_v3, simple_world_comm_v3
"""
import random

import numpy as np
import torch
import torch.nn as nn


# ===========================================================================
# Sub-assignments and configs for each MPE environment
#
# Sub-assignment: maps each position in an agent's observation vector to the
# agent index that "owns" that feature (-1 = self / global features).
# Indexed by AGENT INDEX (not type), one tensor per agent.
#
# Config fields:
#   n_agent_types     — number of distinct observation/action types
#   agent_types       — list of type index for each agent (in env.agents order)
#   other_sub_sizes   — [observer_type][observed_type] = # features visible
#                       per agent of that observed type
#   slfgbl_sub_sizes  — [observer_type] = # self/global features per obs type
#   action_sub_sizes  — [agent_type] = discrete action space size per type
# ===========================================================================

# ---------------------------------------------------------------------------
# simple_v3  (1 agent, obs=4, act=5)
# Single agent navigates to a landmark — no "others" to infer.
# The self-module still learns to predict the agent's own previous action.
# ---------------------------------------------------------------------------
SIMPLE_V3_SUB_ASSIGNMENTS = {
    0: torch.tensor([-1, -1, -1, -1]),
}
SIMPLE_V3_CONFIG = {
    'n_agent_types':    1,
    'agent_types':      [0],
    'other_sub_sizes':  [[0]],
    'slfgbl_sub_sizes': [4],
    'action_sub_sizes': [5],
}

# ---------------------------------------------------------------------------
# simple_spread_v3  (3 cooperative agents, obs=18, act=5)
# Obs per agent: vel(2) + pos(2) + 3*lm_pos(6) + 2*other_pos(4) + 2*comm(4) = 18
# Other-agent section (positions 10-17) interleaves pos then comm for each peer.
# ---------------------------------------------------------------------------
SIMPLE_SPREAD_V3_SUB_ASSIGNMENTS = {
    # agent_0: sees agent_1 at [10-11, 14-15] and agent_2 at [12-13, 16-17]
    0: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      1,  1,  2,  2,  1,  1,  2,  2]),
    # agent_1: sees agent_0 at [10-11, 14-15] and agent_2 at [12-13, 16-17]
    1: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      0,  0,  2,  2,  0,  0,  2,  2]),
    # agent_2: sees agent_0 at [10-11, 14-15] and agent_1 at [12-13, 16-17]
    2: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      0,  0,  1,  1,  0,  0,  1,  1]),
}
SIMPLE_SPREAD_V3_CONFIG = {
    'n_agent_types':    1,
    'agent_types':      [0, 0, 0],
    'other_sub_sizes':  [[4]],   # 2 pos + 2 comm per peer
    'slfgbl_sub_sizes': [10],    # vel(2)+pos(2)+3*lm_pos(6)
    'action_sub_sizes': [5],
}

# ---------------------------------------------------------------------------
# simple_adversary_v3  (adversary_0, agent_0, agent_1; competitive)
# adversary obs=8:  2*lm_pos(4) + 2*other_pos(4)
# agent obs=10:     goal_pos(2) + 2*lm_pos(4) + 2*other_pos(4)
# Agent order:  [adversary_0(0), agent_0(1), agent_1(2)]
# ---------------------------------------------------------------------------
SIMPLE_ADVERSARY_V3_SUB_ASSIGNMENTS = {
    # adversary_0: sees agent_0 at [4-5], agent_1 at [6-7]
    0: torch.tensor([-1, -1, -1, -1,  1,  1,  2,  2]),
    # agent_0: sees adversary_0 at [6-7], agent_1 at [8-9]
    1: torch.tensor([-1, -1, -1, -1, -1, -1,  0,  0,  2,  2]),
    # agent_1: sees adversary_0 at [6-7], agent_0 at [8-9]
    2: torch.tensor([-1, -1, -1, -1, -1, -1,  0,  0,  1,  1]),
}
SIMPLE_ADVERSARY_V3_CONFIG = {
    'n_agent_types':    2,
    'agent_types':      [0, 1, 1],   # adversary=type0, good agent=type1
    'other_sub_sizes':  [[0, 2],     # type0 sees 0 from type0, 2 from type1
                         [2, 2]],    # type1 sees 2 from type0, 2 from type1
    'slfgbl_sub_sizes': [4, 6],      # adv: 2*lm_pos; good: goal+2*lm_pos
    'action_sub_sizes': [5, 5],
}

# ---------------------------------------------------------------------------
# simple_crypto_v3  (eve_0, bob_0, alice_0; obs: 4, 8, 8 — act: 4, 4, 4)
# eve  obs=4:  alice's public message (all assigned global — eve has no self)
# bob  obs=8:  private key(4, global) + alice comm(4, → alice idx=2)
# alice obs=8: goal(4, global) + shared key(4, global)
# Agent order: [eve_0(0), bob_0(1), alice_0(2)]
# ---------------------------------------------------------------------------
SIMPLE_CRYPTO_V3_SUB_ASSIGNMENTS = {
    # eve: all observation is alice's broadcast — treat as global to avoid
    #      slfgbl=0 which would cause empty tensor bugs in self-module
    0: torch.tensor([-1, -1, -1, -1]),
    # bob: key is shared global state; positions 4-7 are alice's comm
    1: torch.tensor([-1, -1, -1, -1,  2,  2,  2,  2]),
    # alice: goal(self) + shared key(global) — nothing to attribute to others
    2: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1]),
}
SIMPLE_CRYPTO_V3_CONFIG = {
    'n_agent_types':    3,
    'agent_types':      [0, 1, 2],   # eve=type0, bob=type1, alice=type2
    'other_sub_sizes':  [[0, 0, 0],  # eve sees 0 attributable to any specific agent
                         [0, 0, 4],  # bob sees 4 from alice (her comm)
                         [0, 0, 0]], # alice sees 0 attributable
    'slfgbl_sub_sizes': [4, 4, 8],   # eve:4, bob: key(4), alice: goal+key(8)
    'action_sub_sizes': [4, 4, 4],
}

# ---------------------------------------------------------------------------
# simple_push_v3  (adversary_0, agent_0; competitive; obs: 8, 19 — act: 5, 5)
# adversary obs=8:  vel(2) + 2*lm_pos(4) + agent_pos(2)
# agent obs=19:     vel(2) + goal_pos(2) + color(3) + 2*lm_pos(4)
#                   + 2*lm_color(6) + adv_pos(2)
# Agent order: [adversary_0(0), agent_0(1)]
# ---------------------------------------------------------------------------
SIMPLE_PUSH_V3_SUB_ASSIGNMENTS = {
    # adversary: positions 6-7 are agent_0's position
    0: torch.tensor([-1, -1, -1, -1, -1, -1,  1,  1]),
    # agent: positions 17-18 are adversary_0's position
    1: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1, -1, -1, -1, -1, -1, -1,  0,  0]),
}
SIMPLE_PUSH_V3_CONFIG = {
    'n_agent_types':    2,
    'agent_types':      [0, 1],      # adversary=type0, agent=type1
    'other_sub_sizes':  [[0, 2],     # type0 sees 2 from type1 (pos)
                         [2, 0]],    # type1 sees 2 from type0 (pos)
    'slfgbl_sub_sizes': [6, 17],     # adv: vel+2*lm_pos; agent: vel+goal+color+2*lm+2*lm_color
    'action_sub_sizes': [5, 5],
}

# ---------------------------------------------------------------------------
# simple_reference_v3  (2 cooperative agents; obs=21, act=50)
# Obs: vel(2) + 3*lm_pos(6) + goal_color(3) + other_agent_comm(10) = 21
# Agent order: [agent_0(0), agent_1(1)]
# ---------------------------------------------------------------------------
SIMPLE_REFERENCE_V3_SUB_ASSIGNMENTS = {
    # agent_0: positions 11-20 are agent_1's communication
    0: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
    # agent_1: positions 11-20 are agent_0's communication
    1: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
}
SIMPLE_REFERENCE_V3_CONFIG = {
    'n_agent_types':    1,
    'agent_types':      [0, 0],
    'other_sub_sizes':  [[10]],   # 10 comm dims from each peer
    'slfgbl_sub_sizes': [11],     # vel(2)+3*lm_pos(6)+goal_color(3)
    'action_sub_sizes': [50],     # large combined action space (move + comm)
}

# ---------------------------------------------------------------------------
# simple_speaker_listener_v4  (speaker_0, listener_0; obs: 3, 11 — act: 3, 5)
# speaker obs=3:   goal_color(3) — all self/global
# listener obs=11: vel(2) + 3*lm_pos(6) + speaker_comm(3)
# Agent order: [speaker_0(0), listener_0(1)]
# ---------------------------------------------------------------------------
SIMPLE_SPEAKER_LISTENER_V4_SUB_ASSIGNMENTS = {
    # speaker: all observations are its own goal color
    0: torch.tensor([-1, -1, -1]),
    # listener: positions 8-10 are speaker_0's communication
    1: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0]),
}
SIMPLE_SPEAKER_LISTENER_V4_CONFIG = {
    'n_agent_types':    2,
    'agent_types':      [0, 1],       # speaker=type0, listener=type1
    'other_sub_sizes':  [[0, 0],      # speaker sees 0 from anyone
                         [3, 0]],     # listener sees 3 comm dims from speaker
    'slfgbl_sub_sizes': [3, 8],       # speaker: goal(3); listener: vel+3*lm_pos(8)
    'action_sub_sizes': [3, 5],
}

# ---------------------------------------------------------------------------
# simple_tag_v3  (3 adversaries, 1 prey; obs: 16/14, act: 5)
# Obs per adversary: vel(2)+pos(2)+vel_of_prey(2)+2*lm_pos(4)+2*other_adv_pos(4)+prey_pos(2) = 16
# Obs of prey:       vel(2)+pos(2)+vel_of_prey(2)+2*lm_pos(4)+2*other_adv_pos(4)           = 14 (no self-pos of adv)
#   Actually: adversary obs has extra prey_vel at positions... let me use confirmed structure:
#   adversary: [-1]*8 slfgbl, then see 2 other-adv positions (2 each), prey pos (4)
#   prey:      [-1]*8 slfgbl, then see 2 adv positions (2 each) + 1 adv pos (2 more)
# Agent order: [adversary_0(0), adversary_1(1), adversary_2(2), agent_0(3)]
# ---------------------------------------------------------------------------
SIMPLE_TAG_SUB_ASSIGNMENTS = {
    0: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  2,  2,  3,  3,  3,  3]),
    1: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  2,  2,  3,  3,  3,  3]),
    2: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  1,  1,  3,  3,  3,  3]),
    3: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  1,  1,  2,  2]),
}
SIMPLE_TAG_CONFIG = {
    'n_agent_types':    2,
    'agent_types':      [0, 0, 0, 1],   # 3 adversaries (type0), 1 prey (type1)
    'other_sub_sizes':  [[2, 4],        # adv sees 2 pos from each adv, 4 from prey
                         [2, 0]],       # prey sees 2 pos from each adv, nothing from prey
    'slfgbl_sub_sizes': [8, 8],
    'action_sub_sizes': [5, 5],
    # slfgbl[0:2]=self_velocity, slfgbl[2:4]=self_absolute_position (both types) --
    # used by --fellow_mode velocity_analytical/velocity_learned to compute
    # delta_pos_i = curr_slfgbl[2:4] - last_slfgbl[2:4].
    'self_pos_offsets': [2, 2],
}

# ---------------------------------------------------------------------------
# simple_world_comm_v3  (leadadv, 3 adv, 2 good; obs: 34/28, act: 20/5/5)
#
# Adversary/leader obs=34:
#   vel(2) + pos(2) + 5*lm_pos(10) + 5*other_pos(10) + 2*good_vel(4)
#   + in_forest(2) + leader_comm(4) = 34
#
# Good agent obs=28:
#   vel(2) + pos(2) + 5*lm_pos(10) + 5*other_pos(10) + in_forest(2)
#   + other_good_vel(2) = 28
#
# Other-agent ordering (for adversaries, skipping self):
#   positions 14-23: the 5 other agents in world.agents order (excluding self)
# positions 24-27 (adversaries): agent_0 vel, agent_1 vel (always good agents)
# positions 28-29 (adversaries): in_forest flags → global (-1)
# positions 30-33 (adversaries): leader comm → global (-1, broadcast info)
#
# Agent order: [leadadversary_0(0), adversary_0(1), adversary_1(2),
#               adversary_2(3), agent_0(4), agent_1(5)]
# ---------------------------------------------------------------------------
SIMPLE_WORLD_COMM_V3_SUB_ASSIGNMENTS = {
    # leadadversary_0 (idx0): other-pos = [adv0(1), adv1(2), adv2(3), agent0(4), agent1(5)]
    # good-vel = [agent0(4), agent1(5)], in_forest=global, own_comm=global
    0: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      1,  1,  2,  2,  3,  3,  4,  4,  5,  5,
                      4,  4,  5,  5,
                     -1, -1, -1, -1, -1, -1]),
    # adversary_0 (idx1): other-pos = [leadadv(0), adv1(2), adv2(3), agent0(4), agent1(5)]
    1: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      0,  0,  2,  2,  3,  3,  4,  4,  5,  5,
                      4,  4,  5,  5,
                     -1, -1, -1, -1, -1, -1]),
    # adversary_1 (idx2): other-pos = [leadadv(0), adv0(1), adv2(3), agent0(4), agent1(5)]
    2: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      0,  0,  1,  1,  3,  3,  4,  4,  5,  5,
                      4,  4,  5,  5,
                     -1, -1, -1, -1, -1, -1]),
    # adversary_2 (idx3): other-pos = [leadadv(0), adv0(1), adv1(2), agent0(4), agent1(5)]
    3: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      0,  0,  1,  1,  2,  2,  4,  4,  5,  5,
                      4,  4,  5,  5,
                     -1, -1, -1, -1, -1, -1]),
    # agent_0 (idx4): other-pos = [leadadv(0), adv0(1), adv1(2), adv2(3), agent1(5)]
    # in_forest=global, other_good_vel=agent_1(5)
    4: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      0,  0,  1,  1,  2,  2,  3,  3,  5,  5,
                     -1, -1,
                      5,  5]),
    # agent_1 (idx5): other-pos = [leadadv(0), adv0(1), adv1(2), adv2(3), agent0(4)]
    # in_forest=global, other_good_vel=agent_0(4)
    5: torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                      0,  0,  1,  1,  2,  2,  3,  3,  4,  4,
                     -1, -1,
                      4,  4]),
}
SIMPLE_WORLD_COMM_V3_CONFIG = {
    'n_agent_types':    3,
    'agent_types':      [0, 1, 1, 1, 2, 2],   # leader=type0, adv=type1, good=type2
    'other_sub_sizes':  [[0, 2, 4],   # leader: 0 from other leaders, 2 from adv, 4 from good
                         [2, 2, 4],   # adv:    2 from leader, 2 from adv, 4 from good
                         [2, 2, 4]],  # good:   2 from leader, 2 from adv, 4 from other good
    'slfgbl_sub_sizes': [20, 20, 16], # leader/adv: vel+pos+5lm+in_forest+leader_comm = 20
                                      # good: vel+pos+5lm+in_forest = 16
    'action_sub_sizes': [20, 5, 5],
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_ENV_REGISTRY = {
    'simple_v3':                 (SIMPLE_V3_SUB_ASSIGNMENTS,                SIMPLE_V3_CONFIG),
    'simple_spread_v3':          (SIMPLE_SPREAD_V3_SUB_ASSIGNMENTS,         SIMPLE_SPREAD_V3_CONFIG),
    'simple_adversary_v3':       (SIMPLE_ADVERSARY_V3_SUB_ASSIGNMENTS,      SIMPLE_ADVERSARY_V3_CONFIG),
    'simple_crypto_v3':          (SIMPLE_CRYPTO_V3_SUB_ASSIGNMENTS,         SIMPLE_CRYPTO_V3_CONFIG),
    'simple_push_v3':            (SIMPLE_PUSH_V3_SUB_ASSIGNMENTS,           SIMPLE_PUSH_V3_CONFIG),
    'simple_reference_v3':       (SIMPLE_REFERENCE_V3_SUB_ASSIGNMENTS,      SIMPLE_REFERENCE_V3_CONFIG),
    'simple_speaker_listener_v4':(SIMPLE_SPEAKER_LISTENER_V4_SUB_ASSIGNMENTS, SIMPLE_SPEAKER_LISTENER_V4_CONFIG),
    'simple_tag_v3':             (SIMPLE_TAG_SUB_ASSIGNMENTS,               SIMPLE_TAG_CONFIG),
    'simple_world_comm_v3':      (SIMPLE_WORLD_COMM_V3_SUB_ASSIGNMENTS,     SIMPLE_WORLD_COMM_V3_CONFIG),
}


def get_env_ptai_config(env_name: str):
    """
    Return (sub_assignments, config) for the given environment.

    Args:
        env_name: PettingZoo MPE environment name

    Returns:
        sub_assignments: dict mapping agent_index -> (obs_dim,) tensor
        config:          dict with n_agent_types, agent_types, other_sub_sizes,
                         slfgbl_sub_sizes, action_sub_sizes
    """
    if env_name not in _ENV_REGISTRY:
        raise ValueError(
            f"PTAI not configured for '{env_name}'. "
            f"Available: {list(_ENV_REGISTRY.keys())}"
        )
    return _ENV_REGISTRY[env_name]


# ===========================================================================
# Building-block network
# ===========================================================================
class Awareness(nn.Module):
    """
    Single module that predicts one agent's previous action from a
    temporally-stacked observation bundle.
    """
    def __init__(self, bundle_size: int, action_sub_size: int, lr: float = 0.005):
        super().__init__()
        self.fc1 = nn.Linear(bundle_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_sub_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, bundle: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(bundle))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def update(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()


class AnalyticalDelta(nn.Module):
    """
    Parameter-free module for fellow-same-type-agent slots in
    `--fellow_mode velocity_analytical`.

    Returns the observed agent's previous-step displacement
        delta_pos_j = delta_relpos_j + delta_pos_i
    computed directly from the bundle's delta segment (the last third of
    cat([curr_bundle, last_bundle, curr_bundle - last_bundle])):
        delta_relpos_j = delta[0 : 2]   (relative position is always the
                                          first 2 dims of the "other" block;
                                          other_sub_size may include trailing
                                          dims such as the fellow's velocity)
        delta_pos_i    = delta[other_sub_size + self_pos_offset : +2]
    """
    def __init__(self, bundle_size: int, other_sub_size: int, self_pos_offset: int = 2):
        super().__init__()
        self.B = bundle_size // 3
        self.other_sub_size = other_sub_size
        self.self_pos_offset = self_pos_offset

    def forward(self, bundle: torch.Tensor) -> torch.Tensor:
        delta = bundle[..., 2 * self.B:]
        d_relpos_j = delta[..., 0:2]
        off = self.other_sub_size + self.self_pos_offset
        d_pos_i = delta[..., off:off + 2]
        return d_relpos_j + d_pos_i

    def update(self, loss: torch.Tensor):
        pass  # parameter-free


# ===========================================================================
# AI_Net manager
# ===========================================================================
class AINet:
    """
    Action Inference Network.

    For observing agent i, produces:
        â_i = concat(onehot(â_{i,1}), ..., onehot(â_{i,N}))
    where â_{i,k} is the predicted previous action of agent k, inferred
    from agent i's current observation o_i and previous observation o_i^-.

    Usage:
        # single sample (select_actions)
        a_hat = ai_net.forward(sub_assigns, observer_ind, curr_obs, last_obs)

        # batch (update loop)
        a_hat = ai_net.forward_batch(sub_assigns, observer_ind, curr_batch, last_batch)
    """

    def __init__(self, n_agent_types, agent_types, other_sub_sizes,
                 slfgbl_sub_sizes, action_sub_sizes, lr=0.005, device='cpu',
                 fellow_mode='action', self_pos_offsets=None):
        self.n_agents         = len(agent_types)
        self.agent_types      = agent_types
        self.device           = device
        self.action_sub_sizes = action_sub_sizes
        self.fellow_mode      = fellow_mode
        self.self_pos_offsets = self_pos_offsets or {}

        # social_modules[observer_type][observed_type] -> Awareness | AnalyticalDelta
        self.social_modules: list[list] = []
        self.self_modules:   list[Awareness] = []
        # social_output_sizes[observer_type][observed_type] -> int
        self.social_output_sizes: list[list[int]] = []

        for observer_type in range(n_agent_types):
            social_set = []
            out_sizes  = []
            for observed_type in range(n_agent_types):
                input_dim = 3 * (other_sub_sizes[observer_type][observed_type]
                                 + slfgbl_sub_sizes[observer_type])
                is_fellow = (observed_type == observer_type)
                if is_fellow and fellow_mode != 'action':
                    out_size = 2
                    if fellow_mode == 'velocity_analytical':
                        mod = AnalyticalDelta(
                            input_dim,
                            other_sub_sizes[observer_type][observed_type],
                            self.self_pos_offsets.get(observer_type, 0)
                        ).to(device)
                    else:  # velocity_learned
                        mod = Awareness(input_dim, out_size, lr).to(device)
                else:
                    out_size = action_sub_sizes[observed_type]
                    mod = Awareness(input_dim, out_size, lr).to(device)
                social_set.append(mod)
                out_sizes.append(out_size)
            self.social_modules.append(social_set)
            self.social_output_sizes.append(out_sizes)
            self.self_modules.append(
                Awareness(3 * slfgbl_sub_sizes[observer_type],
                          action_sub_sizes[observer_type], lr).to(device)
            )

        # Per-observer-type total a_hat length: 1 self-slot + 1 social slot
        # per other agent index, sized via social_output_sizes.
        self.total_action_dim_by_type: dict = {}
        for observer_type in range(n_agent_types):
            rep_ind = agent_types.index(observer_type)
            total = action_sub_sizes[observer_type]  # self-slot
            for observed_ind in range(self.n_agents):
                if observed_ind == rep_ind:
                    continue
                total += self.social_output_sizes[observer_type][agent_types[observed_ind]]
            self.total_action_dim_by_type[observer_type] = total

        # Backward-compat scalar (only correct/used when uniform across types,
        # i.e. fellow_mode == 'action'; diagnostics-only).
        self.total_action_dim = self.total_action_dim_by_type[agent_types[0]]

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def forward(self, sub_assigns: torch.Tensor, observer_ind: int,
                curr_obs: torch.Tensor, last_obs: torch.Tensor) -> torch.Tensor:
        """
        Single-sample forward.

        Args:
            sub_assigns:  (obs_dim,) tensor mapping each feature to an agent index
                          (-1 = self/global)
            observer_ind: integer index of the observing agent
            curr_obs:     (obs_dim,) current observation
            last_obs:     (obs_dim,) previous observation

        Returns:
            (total_action_dim,) predicted joint action vector
        """
        observer_type = self.agent_types[observer_ind]
        slfgbl_mask   = sub_assigns == -1
        curr_slfgbl   = curr_obs[slfgbl_mask]
        last_slfgbl   = last_obs[slfgbl_mask]

        out_subs = []
        for observed_ind in range(self.n_agents):
            if observed_ind == observer_ind:
                bundle = torch.cat([curr_slfgbl, last_slfgbl,
                                    curr_slfgbl - last_slfgbl])
                out_subs.append(self.self_modules[observer_type](bundle))
            else:
                other_mask  = sub_assigns == observed_ind
                curr_other  = curr_obs[other_mask]
                last_other  = last_obs[other_mask]
                curr_bundle = torch.cat([curr_other, curr_slfgbl])
                last_bundle = torch.cat([last_other, last_slfgbl])
                bundle      = torch.cat([curr_bundle, last_bundle,
                                         curr_bundle - last_bundle])
                out_subs.append(
                    self.social_modules[observer_type][self.agent_types[observed_ind]](bundle)
                )

        return torch.cat(out_subs)

    def forward_batch(self, sub_assigns: torch.Tensor, observer_ind: int,
                      curr_obs_batch: torch.Tensor,
                      last_obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass for use during MADDPG gradient updates.

        Args:
            sub_assigns:     (obs_dim,) feature-to-agent mapping
            observer_ind:    integer index of the observing agent
            curr_obs_batch:  (batch, obs_dim)
            last_obs_batch:  (batch, obs_dim)

        Returns:
            (batch, total_action_dim)
        """
        observer_type = self.agent_types[observer_ind]
        slfgbl_mask   = sub_assigns == -1
        curr_slfgbl   = curr_obs_batch[:, slfgbl_mask]
        last_slfgbl   = last_obs_batch[:, slfgbl_mask]

        out_subs = []
        for observed_ind in range(self.n_agents):
            if observed_ind == observer_ind:
                bundle = torch.cat([curr_slfgbl, last_slfgbl,
                                    curr_slfgbl - last_slfgbl], dim=1)
                out_subs.append(self.self_modules[observer_type](bundle))
            else:
                other_mask  = sub_assigns == observed_ind
                curr_other  = curr_obs_batch[:, other_mask]
                last_other  = last_obs_batch[:, other_mask]
                curr_bundle = torch.cat([curr_other, curr_slfgbl], dim=1)
                last_bundle = torch.cat([last_other, last_slfgbl], dim=1)
                bundle      = torch.cat([curr_bundle, last_bundle,
                                         curr_bundle - last_bundle], dim=1)
                out_subs.append(
                    self.social_modules[observer_type][self.agent_types[observed_ind]](bundle)
                )

        return torch.cat(out_subs, dim=1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def eval_mode(self):
        """Set all sub-modules to eval mode (freeze dropout / batch-norm)."""
        for mod in self.self_modules:
            mod.eval()
        for row in self.social_modules:
            for mod in row:
                mod.eval()

    def state_dict(self) -> dict:
        return {
            'self_modules':    [m.state_dict() for m in self.self_modules],
            'social_modules':  [[m.state_dict() for m in row]
                                for row in self.social_modules],
            'agent_types':      self.agent_types,
            'action_sub_sizes': self.action_sub_sizes,
            'total_action_dim': self.total_action_dim,
            'fellow_mode':      self.fellow_mode,
            'social_output_sizes': self.social_output_sizes,
        }

    def load_state_dict(self, state: dict):
        for i, sd in enumerate(state['self_modules']):
            self.self_modules[i].load_state_dict(sd)
        for i, row in enumerate(state['social_modules']):
            for j, sd in enumerate(row):
                self.social_modules[i][j].load_state_dict(sd)


# ===========================================================================
# Helpers
# ===========================================================================
def one_hot(index: int, size: int, device='cpu') -> torch.Tensor:
    vec = torch.zeros(size, device=device)
    vec[int(index)] = 1.0
    return vec


def _make_env(env_name: str, max_episode_length: int):
    """Create the PettingZoo parallel env for the given environment name."""
    from pettingzoo.mpe.simple import simple as simple_v3
    from pettingzoo.mpe.simple_adversary import simple_adversary as simple_adversary_v3
    from pettingzoo.mpe.simple_crypto import simple_crypto as simple_crypto_v3
    from pettingzoo.mpe.simple_push import simple_push as simple_push_v3
    from pettingzoo.mpe.simple_reference import simple_reference as simple_reference_v3
    from pettingzoo.mpe.simple_speaker_listener import simple_speaker_listener as simple_speaker_listener_v4
    from pettingzoo.mpe.simple_spread import simple_spread as simple_spread_v3
    from pettingzoo.mpe.simple_tag import simple_tag as simple_tag_v3
    from pettingzoo.mpe.simple_world_comm import simple_world_comm as simple_world_comm_v3

    env_map = {
        'simple_v3':                  simple_v3,
        'simple_adversary_v3':        simple_adversary_v3,
        'simple_crypto_v3':           simple_crypto_v3,
        'simple_push_v3':             simple_push_v3,
        'simple_reference_v3':        simple_reference_v3,
        'simple_speaker_listener_v4': simple_speaker_listener_v4,
        'simple_spread_v3':           simple_spread_v3,
        'simple_tag_v3':              simple_tag_v3,
        'simple_world_comm_v3':       simple_world_comm_v3,
    }
    if env_name not in env_map:
        raise ValueError(f"Unknown environment for PTAI: {env_name}")
    return env_map[env_name].parallel_env(max_cycles=max_episode_length)


# ===========================================================================
# Pre-training (Algorithm 2)
# ===========================================================================
def pre_train_ai_net(env_name: str,
                     episode_count: int,
                     max_episode_length: int,
                     percent_to_train_on: float,
                     lr: float = 0.001,
                     device='cpu',
                     verbose: bool = True,
                     fellow_mode: str = 'action'):
    """
    Pre-train AI_Net on random-action episodes for the given environment.

    Args:
        env_name:             PettingZoo MPE environment name
        episode_count:        Number of random-action episodes to train on
        max_episode_length:   Max steps per episode
        percent_to_train_on:  Fraction of (observer, observed) pairs to train each step
        lr:                   Learning rate for Awareness modules
        device:               'cpu' or 'cuda'
        verbose:              Print progress every 10 episodes

    Returns:
        ai_net (AINet):           trained and frozen network
        sub_assignments (dict):   agent_index -> sub_assign tensor
    """
    sub_assignments, cfg = get_env_ptai_config(env_name)

    env    = _make_env(env_name, max_episode_length)
    agents = env.possible_agents   # fixed ordering throughout
    env.reset()  # initialise env state (return value unused here)

    self_pos_offsets = dict(enumerate(cfg.get('self_pos_offsets', [0] * cfg['n_agent_types'])))

    ai_net = AINet(
        cfg['n_agent_types'], cfg['agent_types'],
        cfg['other_sub_sizes'], cfg['slfgbl_sub_sizes'],
        cfg['action_sub_sizes'], lr=lr, device=device,
        fellow_mode=fellow_mode, self_pos_offsets=self_pos_offsets,
    )

    for episode in range(episode_count):
        observations, _   = env.reset()
        last_observations = None
        last_actions      = None

        while env.agents:
            # Sample random actions for all currently active agents
            actions = {agent: env.action_space(agent).sample()
                       for agent in env.agents}

            next_observations, _, _, _, _ = env.step(actions)

            # Train on (last_obs → curr_obs, action) tuples
            if last_observations is not None:
                for observer_ind, agent_name in enumerate(agents):
                    if (agent_name not in observations
                            or agent_name not in last_observations):
                        continue

                    observer_type = cfg['agent_types'][observer_ind]
                    sub_assign    = sub_assignments[observer_ind]

                    curr_obs = torch.tensor(
                        observations[agent_name], dtype=torch.float32
                    ).to(device)
                    last_obs = torch.tensor(
                        last_observations[agent_name], dtype=torch.float32
                    ).to(device)

                    slfgbl_mask = sub_assign == -1
                    curr_slfgbl = curr_obs[slfgbl_mask]
                    last_slfgbl = last_obs[slfgbl_mask]

                    for observed_ind, observed_name in enumerate(agents):
                        if observed_name not in last_actions:
                            continue
                        if random.random() > percent_to_train_on:
                            continue

                        observed_type = cfg['agent_types'][observed_ind]

                        if observed_ind == observer_ind:
                            action_size = cfg['action_sub_sizes'][observed_type]
                            target = one_hot(last_actions[observed_name], action_size, device=device)
                            bundle = torch.cat([curr_slfgbl, last_slfgbl,
                                                curr_slfgbl - last_slfgbl])
                            a_pred = ai_net.self_modules[observer_type](bundle)
                            loss   = nn.MSELoss()(a_pred, target)
                            ai_net.self_modules[observer_type].update(loss)
                        else:
                            other_mask  = sub_assign == observed_ind
                            curr_other  = curr_obs[other_mask]
                            last_other  = last_obs[other_mask]
                            curr_bundle = torch.cat([curr_other, curr_slfgbl])
                            last_bundle = torch.cat([last_other, last_slfgbl])
                            delta_bundle = curr_bundle - last_bundle
                            bundle      = torch.cat([curr_bundle, last_bundle, delta_bundle])

                            is_fellow = (observed_type == observer_type)
                            module = ai_net.social_modules[observer_type][observed_type]

                            if is_fellow and fellow_mode != 'action':
                                other_sub_size = cfg['other_sub_sizes'][observer_type][observer_type]
                                self_pos_off   = self_pos_offsets[observer_type]
                                d_relpos_j = delta_bundle[0:2]
                                off = other_sub_size + self_pos_off
                                target = d_relpos_j + delta_bundle[off:off + 2]
                            else:
                                action_size = cfg['action_sub_sizes'][observed_type]
                                target = one_hot(last_actions[observed_name], action_size, device=device)

                            a_pred = module(bundle)
                            if not (is_fellow and fellow_mode == 'velocity_analytical'):
                                loss = nn.MSELoss()(a_pred, target)
                                module.update(loss)

            last_observations = {k: v.copy() for k, v in observations.items()}
            last_actions      = dict(actions)
            observations      = next_observations

        if verbose and (episode + 1) % 10 == 0:
            print(f"  AI_Net pre-training [{env_name}]: {episode + 1}/{episode_count} episodes")

    env.close()
    ai_net.eval_mode()
    return ai_net, sub_assignments


# ===========================================================================
# CLI entry-point: python PTAI.py [--env simple_tag_v3]
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-train AI_Net for an MPE environment")
    parser.add_argument('--env', type=str, default='simple_tag_v3',
                        choices=list(_ENV_REGISTRY.keys()),
                        help='Environment to pre-train on')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max_len',  type=int, default=20)
    parser.add_argument('--pct',      type=float, default=0.8)
    parser.add_argument('--lr',       type=float, default=0.001)
    cli = parser.parse_args()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pre-training AI_Net for {cli.env} on {_device} ...")
    _ai_net, _sub_assignments = pre_train_ai_net(
        env_name=cli.env,
        episode_count=cli.episodes,
        max_episode_length=cli.max_len,
        percent_to_train_on=cli.pct,
        lr=cli.lr,
        device=_device,
    )
    out_path = f'AI_Net_{cli.env}.pt'
    torch.save(
        {'ai_net': _ai_net.state_dict(), 'sub_assignments': _sub_assignments},
        out_path
    )
    print(f"Saved {out_path}")
