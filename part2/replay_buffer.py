from collections import deque, namedtuple
import random 



class ReplayBuffer:
    def __init__(self, max_size) -> None:
        
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.step = namedtuple("Step", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.buffer)
    
    def __str__(self):
        print(f"Replay buffer filled with {self.__len__()} steps")

    def clear(self):
        self.buffer.clear()

    def append(self, state, action, reward, next_state, done):
        s = self.step(state, action, reward, next_state, done)
        self.buffer.append(s)
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        samples = random.sample(self.buffer, k=batch_size)
        for sample in samples:
            states.append(sample.state)
            actions.append(sample.action)
            rewards.append(sample.reward)
            next_states.append(sample.next_state)
            dones.append(sample.dones)
        
        return states, actions, rewards, next_states, dones 