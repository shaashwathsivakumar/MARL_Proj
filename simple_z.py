import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pettingzoo.mpe import simple_tag_v3
from train_MADDPG import MADDPGAgent
#from train_MADDPG import MADDPG
#from main import get_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v3', 'simple_spread_v3', 'simple_tag_v3'])
    parser.add_argument('folder', type=str, help='name of the folder where model is saved')
    parser.add_argument('--episode-num', type=int, default=1000, help='total episode num during evaluation')
    parser.add_argument('--episode-length', type=int, default=50, help='steps per episode')

    args = parser.parse_args()

    model_dir = os.path.join('./results', args.env_name, args.folder)
    assert os.path.exists(model_dir)
    gif_dir = os.path.join(model_dir, 'gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif

    env = simple_tag_v3.parallel_env(render_mode="rgb_array", max_cycles= args.episode_length)
    obs, info = env.reset()


    agents = {agent: MADDPGAgent(len(obs[agent]), env.action_space(agent).n, len(env.state()), sum(env.action_space(a).n for a in env.agents), 0.001, torch.device("cuda" if torch.cuda.is_available() else "cpu")) for agent in env.agents}


    #maddpg = MADDPG.load(dim_info, os.path.join(model_dir, 'model.pt'))

    agent_num = len(agents.items())

    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    for episode in range(args.episode_num):
        observations, info = env.reset()
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        frame_list = []  # used to save gif
        while env.agents:  # interact with the env for an episode
            actions = {}
            for agent in agents:
                action = agents[agent].get_action(observations[agent], deterministic=True)
                actions[agent] = action
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            frame_list.append(Image.fromarray(env.render()))
            observations = next_obs

            for agent_id, reward in rewards.items():  # update reward
                agent_reward[agent_id] += reward

        env.close()
        message = f'episode {episode + 1}, '
        # episode finishes, record reward
        for agent_id, reward in agent_reward.items():
            episode_rewards[agent_id][episode] = reward
        #    message += f'{agent_id}: {reward:>4f}; '
        #print(message)
        # save gif
        frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                           save_all=True, append_images=frame_list[1:], duration=1, loop=0)

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent_id)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    total_files = len([file for file in os.listdir(model_dir)])
    title = f'evaluate result of maddpg solve {args.env_name} {total_files - 3}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))