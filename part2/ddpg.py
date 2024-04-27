from actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer
from ou_noise import OUNoise
import torch
from torch import optim
from torch import nn


class DDPGAgent():
    def __init__(self, batch_size=64, gamma=0.95, tau=1e-3, actor_lr=1e-4, critic_lr=1e-3, epsilon_start=1, epsilon_end=0.05, epsilon_decay=1e-6, n_time_steps=10_000, n_learn_updates=3):
        
        # Device
        if torch.cuda.is_available(): 
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Actor
        self.actor = Actor(device=self.device)
        self.target_actor = Actor(device=self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_lr = actor_lr
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Critic
        self.critic = Critic(device=self.device)
        self.target_critic = Critic(device=self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_lr = critic_lr
        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Noise
        self.ou_noise = OUNoise(size=(2,), seed=0) 
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Replay Buffer
        self.buffer = ReplayBuffer(1_000_000)
        self.batch_size = batch_size

        # Algorithm parameters
        self.gamma = gamma
        self.tau = tau
        self.n_time_steps = n_time_steps # number of time steps before updating network parameters
        self.n_learn_updates = n_learn_updates # number of updates per learning step
    

    def preprocess_state(self, state):

        processed_state = state
        if isinstance(state, tuple):
            processed_state = state[0]
        
        processed_state = torch.tensor(processed_state)

        if processed_state.dtype == torch.uint8:
            processed_state = processed_state.type(torch.float32)/255.

        if processed_state.dim() == 3:
            processed_state = processed_state.unsqueeze(0)
        
        return processed_state


    def get_action(self, state, eval=False):
        
        action = self.actor.forward(state.to(device=self.device))

        if not eval:
            action += self.ou_noise.sample().to(device=self.device) * self.epsilon

        action = torch.clip(action, -1, 1)

        return action.cpu().detach().numpy()[0]
    
    def reset(self):
        self.ou_noise.reset()

    def save(self, state, action, reward, next_state, done):
        self.buffer.append(state, action, reward, next_state, done)
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay

    def train_policy(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)  
        states, actions, rewards, next_states, dones = states.to(device=self.device), actions.to(device=self.device), rewards.to(device=self.device), next_states.to(device=self.device), dones.to(device=self.device)

        next_actions = self.target_actor.forward(next_states)
        target_q_values = rewards + self.gamma * (1-dones) * self.target_critic.forward(next_states, next_actions)


        # Critic training
        self.critic.zero_grad()

        pred_q_values = self.critic.forward(states, actions)
        critic_loss = self.critic_criterion(pred_q_values, target_q_values).type(torch.FloatTensor)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor training
        self.actor.zero_grad()

        pred_actions = self.actor.forward(states)
        actor_loss = -torch.mean(self.critic.forward(states, pred_actions))
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update 
        for critic_params, target_critic_params in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_critic_params.data.copy_(self.tau * critic_params.data + (1.0 - self.tau) *  target_critic_params.data)
        
        for actor_params, target_actor_params in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_params.data.copy_(self.tau * actor_params.data + (1.0 - self.tau) *  target_actor_params.data)
