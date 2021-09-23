import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return  x

class Reinforce(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate, gamma, device):
        super(Reinforce, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)

        self.device = device

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.gamma = gamma

    def act(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs =  self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs) # create a action distribution
        action = action_dist.sample() # then random sample
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0 # return

        self.optimizer.zero_grad()
        
        # compute return
        for i in reversed(range(len(reward_list))):
            # from back to forth
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            
            log_prob = torch.log(self.policy_net(state).gather(1, action))

            G = self.gamma*G + reward
            loss = - log_prob * G 

            loss.backward() # the gradient can be cummulated
            
        self.optimizer.step()
    
