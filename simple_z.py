from pettingzoo.mpe import simple_tag_v3

env = simple_tag_v3.parallel_env(render_mode="human", max_cycles=1000)
observations, infos = env.reset()
i = 0

while env.agents:
    
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(f"i: {i}, obs: {observations}")
    i += 1
    
env.close()

#Step 1: Know how to interpret each state
#Step 2: Know how to interpret each action
#Step 3: Create a policy that takes in the state and outputs an action
#Step 4: Put this all in a GitHub repo so that everyone can have full access to it.
