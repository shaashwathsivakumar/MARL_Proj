from pettingzoo.mpe import simple_tag_v3

env = simple_tag_v3.parallel_env(render_mode="human", max_cycles=1000)
observations, infos = env.reset()

# this works
print(env.num_agents)

# but this doesn't
print(env.unwrapped.num_adversaries)

# so that means the attributes don't come from the scenario class, but where does num_agents come from?