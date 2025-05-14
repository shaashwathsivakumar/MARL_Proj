import torch
from pettingzoo.mpe import simple_tag_v3
from train_MADDPG import Actor, Critic, ReplayBuffer
import torch.nn as nn


soft = nn.Softmax(dim=0)

env = simple_tag_v3.parallel_env(render_mode="human", max_cycles=1000)
# just get the environment shape
observations, infos = env.reset()
adversary_list = env.agents[0:-1]
actors = [Actor(len(observations[adversary]), 5) for adversary in adversary_list]
critics = [Critic(len(env.state()), env.action_space(adversary).n) for adversary in adversary_list]
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
    actions[adversary] = torch.multinomial(soft(actors[i].forward(observations[adversary])), num_samples=1).item()
next_observations, rewards, terminations, truncations, infos = env.step(actions)
next_state = env.state()
D.push(state, actions, rewards, next_state, observations)
observations = next_observations

if len(D.buffer) >= minibatch_size:
    for i, adversary in enumerate(adversary_list):
        minibatch = D.sample(minibatch_size)
        targets = []
        evals = []
        for s_state, s_actions, s_rewards, s_next_state, s_observations in minibatch:
            s_next_actions = {"agent_0": env.action_space("agent_0").sample()}
            for j, each_adversary in enumerate(adversary_list):
                obs_tensor = torch.tensor(s_observations[each_adversary], dtype=torch.float32)
                s_next_actions[each_adversary] = torch.multinomial(soft(actors[j].forward(obs_tensor)), num_samples=1).item()
            reward = torch.tensor([s_rewards[adversary]], dtype=torch.float32)
            targets.append(reward + gamma * critics[i].forward(s_next_state, s_next_actions).detach())
            evals.append(critics[i].forward(s_state, s_actions))

        loss = torch.nn.functional.mse_loss(torch.stack(targets).squeeze(), torch.stack(evals).squeeze())
        critic_optimizers[i].zero_grad()
        loss.backward()
        critic_optimizers[i].step()




