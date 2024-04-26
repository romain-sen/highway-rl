import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(4, 4, 5, 2) # no padding as car in middle of image
        self.conv2 = nn.Conv2d(4, 8, 5, 2, padding=2) # kernel_size = 5
        # self.drop = nn.Dropout2d()
        self.fc_1 = nn.Linear(1800, 256)   
        self.fc_2 = nn.Linear(256, 2)
    
    def forward(self, state):
        assert state.dim() == 4 # (BS, C, W, H)

        x = F.relu(self.conv1(state))
        # x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x,2,2)
        x = x.view(-1, (1800))
        x = F.relu(self.fc_1(x))
        x = F.tanh(self.fc_2(x))

        return x


class Critic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Action head
        self.act_fc1 = nn.Linear(2, 16)
        self.act_fc2 = nn.Linear(16, 16)

        # State head
        self.conv1 = nn.Conv2d(4, 4, 5, 2) # no padding as car in middle of image
        self.conv2 = nn.Conv2d(4, 8, 5, 2, padding=2) # kernel_size = 5
        # self.drop = nn.Dropout2d()
        self.fc_1 = nn.Linear(1800, 256)   
        self.fc_2 = nn.Linear(256, 32)

        # Output network
        self.out_fc1 = nn.Linear(48, 128)
        self.out_fc2 = nn.Linear(128, 128)
        self.out_fc3 = nn.Linear(128, 1)


    
    def forward(self, state, action):
        assert state.dim() == 4 # (BS, C, W, H)
        assert action.dim() == 2 # (BS, A)

        # State
        state_embed = F.relu(self.conv1(state))
        # x = F.max_pool2d(x,2,2)
        state_embed = F.relu(self.conv2(state_embed))
        # x = F.max_pool2d(x,2,2)
        state_embed = state_embed.view(-1, (1800))
        state_embed = F.relu(self.fc_1(state_embed))
        state_embed = F.tanh(self.fc_2(state_embed))
        # print(state_embed.shape)

        #Action
        action_embed = F.relu(self.act_fc1(action))
        action_embed = F.relu(self.act_fc2(action_embed))
        # action_embed = action_embed.unsqueeze(-2)
        # print(action_embed.shape)

        # Output
        out = torch.concat([action_embed, state_embed], axis=-1)
        out = F.relu(self.out_fc1(out))
        out = F.relu(self.out_fc2(out))
        out = self.out_fc3(out)

        return out


