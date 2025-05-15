import torch
from pettingzoo.mpe import simple_tag_v3
from train_MADDPG import Actor, Critic, ReplayBuffer, one_hot
import torch.nn as nn


soft = nn.Softmax(dim=-1)

env = simple_tag_v3.parallel_env(render_mode="human", max_cycles=1000)
# just get the environment shape
observations, infos = env.reset()
adversary_list = env.agents[0:-1]
actors = [Actor(len(observations[adversary]), 5) for adversary in adversary_list]
critics = [Critic(len(env.state()), env.action_space(adversary).n * len(adversary_list)) for adversary in adversary_list]
actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.001) for actor in actors]
critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=0.001) for critic in critics]
D = ReplayBuffer(100)
gamma = 0.9
minibatch_size = 1

# iterate over this many episodes
observations, infos = env.reset()
# iterate over this many steps
state = env.state()
actions = {"agent_0": env.action_space("agent_0").sample()}
for i, adversary in enumerate(adversary_list):
    actions[adversary] = torch.multinomial(soft(actors[i](torch.tensor(observations[adversary]))), num_samples=1).item()
next_observations, rewards, terminations, truncations, infos = env.step(actions)
next_state = env.state()
D.push(state, actions, rewards, next_state, observations)
observations = next_observations

if len(D.buffer) >= minibatch_size:
    for i, adversary in enumerate(adversary_list):
        minibatch = D.sample(minibatch_size)
        targets = []
        evals = []
        actor_losses = []
        for s_state, s_actions, s_rewards, s_next_state, s_observations in minibatch:
            # critic loss
            s_next_actions = []
            for j, each_adversary in enumerate(adversary_list):
                obs_tensor = torch.tensor(s_observations[each_adversary], dtype=torch.float32)
                s_next_actions.append(one_hot(torch.multinomial(soft(actors[j](obs_tensor)), num_samples=1).item(), env.action_space(each_adversary).n))
            reward = torch.tensor([s_rewards[adversary]], dtype=torch.float32)
            s_next_state_t = torch.tensor(s_next_state, dtype=torch.float32)
            targets.append(reward + gamma * critics[i](s_next_state_t.detach().clone(), torch.cat(s_next_actions, dim=-1).detach()))
            s_state_t = torch.tensor(s_state, dtype=torch.float32)
            evals.append(critics[i](s_state_t.detach().clone(), torch.cat([one_hot(s_actions[ag], env.action_space(ag).n) for ag in adversary_list], dim=-1).detach()))
            # actor loss
            joint_actions = []
            for j, each_adversary in enumerate(adversary_list):
                obs_j = torch.tensor(s_observations[each_adversary], dtype=torch.float32)
                action_probs_j = actors[j](obs_j)
                if j == i:
                    joint_actions.append(action_probs_j)
                else:
                    detached_action = action_probs_j.detach()
                    detached_action.requires_grad = False  # ensure it's not tracked
                    joint_actions.append(detached_action)
            joint_action_vector = torch.cat(joint_actions, dim=-1)
            s_state_tensor = torch.tensor(s_state, dtype=torch.float32)

            # Check if state requires gradient
            print(f"[DEBUG] s_state_tensor.requires_grad: {s_state_tensor.requires_grad}")
            print(f"[DEBUG] s_state_tensor shape: {s_state_tensor.shape}")
            # Check if joint_action_vector requires gradient
            print(f"[DEBUG] joint_action_vector.requires_grad: {joint_action_vector.requires_grad}")
            print(f"[DEBUG] joint_action_vector shape: {joint_action_vector.shape}")
            # Inspect each part of the joint action to verify only the current agent's action has requires_grad = True
            action_sizes = [env.action_space(agent).n for agent in adversary_list]
            start = 0
            for j, (agent, size) in enumerate(zip(adversary_list, action_sizes)):
                action_slice = joint_actions[j]
                print(
                    f"[DEBUG] Agent {j} ({'current' if j == i else 'other'}) action requires_grad: {action_slice.requires_grad}")
                start += size
            # Optionally test critic output computation
            critic_output = critics[i](s_state_tensor, joint_action_vector)
            print(f"[DEBUG] critic_output requires_grad: {critic_output.requires_grad}")
            with torch.autograd.detect_anomaly():
                actor_loss = -critics[i](s_state_tensor, joint_action_vector)
            actor_losses.append(-critics[i](s_state_tensor, joint_action_vector))
        # critic loss
        critic_loss = torch.nn.functional.mse_loss(torch.stack(targets).squeeze(), torch.stack(evals).squeeze())
        actor_optimizers[i].zero_grad()
        critic_optimizers[i].zero_grad()
        critic_loss.backward()
        critic_optimizers[i].step()
        # actor loss
        actor_loss = torch.stack(actor_losses).mean()
        actor_optimizers[i].zero_grad()
        critic_optimizers[i].zero_grad()
        with torch.autograd.detect_anomaly():
            actor_loss.backward()
        actor_optimizers[i].step()




