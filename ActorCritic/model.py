import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import collections
import random

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

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim,
                actor_lr, critic_lr,
                gamma, device):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.device = device

        self.actor = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # self.replay_buffer = ReplayBuffer(max_capacity)

    def act(self, state):

        state = torch.tensor([state], dtype=torch.float).to(self.device)
        
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()

        return action.item()
    
    def update(self, transition_dict):

        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        
        td_error = td_target - self.critic(states) # used in actor

        log_probs = torch.log(self.actor(states).gather(1, actions))
        # actor_loss = torch.mean(-log_probs*self.critic(states).detach())
        actor_loss = torch.mean(-log_probs * td_error.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()
        # print('td error', td_error)
        # print('actor loss:', actor_loss.item())
        # print('critic loss:', critic_loss.item())
        self.actor_optimizer.step()
        self.critic_optimizer.step()


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