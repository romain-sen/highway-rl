import highway_env
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from ddpg import DDPGAgent
from config import config
from collections import deque
import numpy as np
import torch
import os
print(os.path.abspath('.'))

env = gym.make("racetrack-v0", render_mode="rgb_array")
env.unwrapped.configure(config)

ddpg_agent = DDPGAgent()
ddpg_agent.reset()

device = ddpg_agent.device

checkpoint_actor = torch.load("git/highway-rl/part2/trained_policy/actor.h5", map_location=device)
ddpg_agent.actor.load_state_dict(checkpoint_actor)

NB_EPISODES = 5
for n_iter in tqdm(range(1, NB_EPISODES+1)):
    score = 0
    t = 0
    state = env.reset()
    state = ddpg_agent.preprocess_state(state)
    done = False
    while not done:
        env.render()
        action = ddpg_agent.get_action(state, eval=True)
        next_state, reward, done, truncated, info = env.step(action)
        if not info['rewards']['on_road_reward']:
            done = True
            reward = -1
        done = int(done or truncated) 
        state = ddpg_agent.preprocess_state(next_state)
        score += reward
        t += 1
    
    print(f"Episode {n_iter} done in {t} steps, score: {score}")
