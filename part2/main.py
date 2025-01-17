# from sac import SACAgent
import highway_env
import gymnasium as gym
import matplotlib.pyplot as plt
# from config import config
from highway_env.envs import IntersectionEnv
from tqdm import tqdm
from actor_critic import *
from torchsummary import summary
from ddpg import DDPGAgent
from config import config
from collections import deque
import numpy as np
import os


# actor = Actor()
# action = actor.forward(torch.zeros((2, 4, 64, 64)))
# critic = Critic()
# score = critic.forward(torch.zeros((2, 4, 64, 64)), torch.zeros((2, 2)))

# summary(actor, (4, 64, 64))
# print('\n')
# summary(critic, [(4, 64, 64), (2,)])

env = gym.make("racetrack-v0", render_mode="rgb_array")
env.unwrapped.configure(config)

SAVE_PATH = "RL/highway-rl/part2/trained_policy/"
NB_EPISODES = 10_000 
average_end_score = 300.

ddpg_agent = DDPGAgent()
ddpg_agent.reset()
device = ddpg_agent.device

checkpoint_actor = torch.load(os.path.join(SAVE_PATH, "actor_20000.h5"), map_location=device)
ddpg_agent.actor.load_state_dict(checkpoint_actor)
checkpoint_critic = torch.load(os.path.join(SAVE_PATH, "critic_20000.h5"), map_location=device)
ddpg_agent.critic.load_state_dict(checkpoint_critic)

scores_window = deque(maxlen=50)
scores = []
mean_scores = []

for n_iter in tqdm(range(1, NB_EPISODES+1)):
    ddpg_agent.reset()
    score = 0
    t = 0
    state = env.reset()
    done = not state[1]['rewards']['on_road_reward']
    state = ddpg_agent.preprocess_state(state)

    while not done:
        action = ddpg_agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        if not info['rewards']['on_road_reward'] or info['speed']<0 : # for racetrack-v0 env
            done = True
            reward = -1
        
        speed = min(10, info['speed'])
        reward += 0.1*speed - 0.5 

        done = int(done or truncated) 
        next_state = ddpg_agent.preprocess_state(next_state)
        ddpg_agent.save(state, action, reward, next_state, done)
        state = next_state
        score += reward
        t += 1
    
    scores_window.append(score)
    scores.append(score)
    mean_scores.append(np.mean(scores_window))

    if len(ddpg_agent.buffer) >= ddpg_agent.batch_size:
        ddpg_agent.train_policy()
        ddpg_agent.update_epsilon()

    if n_iter % 100 == 0:
        print(f'Episode {n_iter}\tAverage Score: {np.mean(scores_window)}')
        torch.save(ddpg_agent.actor.state_dict(), os.path.join(SAVE_PATH, 'actor.h5'))
        torch.save(ddpg_agent.critic.state_dict(), os.path.join(SAVE_PATH, 'critic.h5'))
        plt.clf()
        plt.plot(np.arange(len(scores)), scores, label="Score")
        plt.plot(np.arange(len(mean_scores)), mean_scores, label="Mean Score")
        plt.legend()
        plt.xlabel('# episodes')
        plt.ylabel('Score')
        plt.title("Mean scores over training")
        plt.savefig(os.path.join(SAVE_PATH, "mean_scores_training.png"))
        
    if np.mean(scores_window) >= average_end_score:
        print(f'Environment solved in {n_iter} episodes.\tAverage Score: {np.mean(scores_window)}')
        torch.save(ddpg_agent.actor.state_dict(), os.path.join(SAVE_PATH, 'final_actor.h5'))
        torch.save(ddpg_agent.critic.state_dict(), os.path.join(SAVE_PATH, 'final_critic.h5'))
        plt.clf()
        plt.plot(np.arange(len(scores)), scores, label="Score")
        plt.plot(np.arange(len(mean_scores)), mean_scores, label="Mean Score")
        plt.legend()
        plt.xlabel('# episodes')
        plt.ylabel('Score')
        plt.title("Mean scores over training")
        plt.savefig(os.path.join(SAVE_PATH, "mean_scores_training.png"))
        break
            
env.close()



raise