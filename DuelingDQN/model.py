import torch
import torch.nn as nn
import collections
import random
import torch.nn.functional as F
import numpy as np

class ReplayBuffer:
    def __init__(self, max_capacity:int):
        self.buffer = collections.deque(maxlen=max_capacity)


    def add(self, state, action, reward, next_state, done):
        """
        record order: [s,a,r,s,done]
        """
        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch_size:int):

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


    def __len__(self):
        return len(self.buffer)



class DuelingDQNAgent(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=64):
        super(DuelingDQNAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, hidden_dim) # shared part

        self.fcA = nn.Linear(hidden_dim, action_dim) # advantage layer
        self.fcV = nn.Linear(hidden_dim, 1) # value layer
    
    def forward(self, x):

        advantage = self.fcA(F.relu(self.fc1(x)))
        value = self.fcV(F.relu(self.fc1(x)))

        Q = value + advantage - advantage.mean(-1).view(-1,1)
        return Q


class DuelingDQN(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int, 
                learning_rate, gamma, epsilon, target_update,
                max_capacity,
                device):

        super(DuelingDQN, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.device = device
        self.q_net = DuelingDQNAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        self.target_q_net = DuelingDQNAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
        
        self.optimizer= torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.gamma = gamma # discount factor
        self.epsilon = epsilon # epsilon greedy

        self.target_update = target_update

        self.count = 0

        self.weight_init()

        self.replay_buffer = ReplayBuffer(max_capacity)


    def weight_init(self):
        for param in self.q_net.parameters():
            nn.init.uniform_(param)
        # self.target_q_net.load_state_dict(self.q_net.state_dict()) # keep the same
        for param in self.target_q_net.parameters():
            nn.init.uniform_(param)

    def act(self, state):
        if np.random.random() < self.epsilon: # or epsilon = self.epsilon*(1./(self.count+1))
            action = np.random.randint(self.action_dim)
        else:
            # import pdb; pdb.set_trace()
            if state.__class__==np.ndarray:
                state = torch.tensor(state, dtype=torch.float)
            state = state.to(self.device)
            action = torch.argmax(self.q_net(state))
            action = action.cpu().numpy()
        return action

    def update(self, transition_dict):
        # get transition batch
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)

        actions =  torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # computer Q
        q_values = self.q_net(states).gather(1, actions) # 1 means operating on each single transition

        max_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)

        q_targets = rewards + self.gamma * max_q_values * (1 - dones)
        

        # mean squared error loss
        loss = torch.mean(F.mse_loss(q_values, q_targets))


        # update q net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target q net
        # Hard update
        if self.count % self.target_update==0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
        
    def record(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)


    def sample(self, batch_size:int):
        if batch_size>len(self.replay_buffer):
            print('Not enough samples in replay buffer.')
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        transitions = dict()
        transitions['states'] = states
        transitions['actions'] = actions
        transitions['rewards'] = rewards
        transitions['next_states'] = next_states
        transitions['dones'] = dones

        return transitions

