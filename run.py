import argparse
import gym
import torch
import utils
import numpy as np


# our benchmark
from DQN import DQN
from DoubleDQN import DoubleDQN


def main(args):

    print(args)

    # set seed (for reproducibility)
    seed = args.seed
    utils.setup_seed(seed)

    # set env
    env_name = args.env
    env = gym.make(env_name)
    env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # set agent
    agent_name = args.benchmark
    agent = None
    gamma = args.gamma
    epsilon = args.epsilon
    device = args.device
    hidden_dim = args.hidden
    lr = args.lr
    if agent_name=='DQN':
        target_update = args.target_update
        agent = DQN(state_dim=state_dim, action_dim=action_dim,
                    hidden_dim=hidden_dim, learning_rate=lr,
                    gamma=gamma, epsilon=epsilon,
                    target_update=target_update,
                    max_capacity=args.max_capacity,
                    device=device).float()
    elif agent_name=='DoubleDQN':
        target_update = args.target_update
        agent = DoubleDQN(state_dim=state_dim, action_dim=action_dim,
                    hidden_dim=hidden_dim, learning_rate=lr,
                    gamma=gamma, epsilon=epsilon,
                    target_update=target_update,
                    max_capacity=args.max_capacity,
                    device=device).float()
    else:
        raise Exception('%s has not been implemented yet.'%agent_name)
    

    # running!
    epoch_num = args.epoch
    batch_size = args.batch_size
    reward_sum_list = []

    for epoch in range(epoch_num):

        cur_state = env.reset()
        cur_state = np.array(cur_state, dtype=np.float)
        done = False

        reward_sum = 0 # record

        while(not done):
            action = agent.act(cur_state)
            next_state,reward, done, _  = env.step(action)
            next_state = np.array(next_state, dtype=np.float)
            
            # DQN, DoubleDQN
            if agent_name=='DQN' or agent_name=='DoubleDQN' :

                agent.record(cur_state, action, reward, next_state, done)
                if len(agent.replay_buffer) > batch_size:
                    transitions = agent.sample(batch_size)
                    agent.update(transitions)
                else:
                    continue
            
            cur_state = next_state # update state
            reward_sum += reward
            if done:
                break

        if args.plot:
            reward_sum_list.append(reward_sum)
        print('epoch %u \'s reward sum: %f'%(epoch, reward_sum))

    if args.save:
        utils.save_model(model_name='%s_%s'%(agent_name,env_name), model=agent)

    if args.plot:
        utils.plot_one(model_name=agent_name, env_name=env_name, reward_list=reward_sum_list)



# parser
parser = argparse.ArgumentParser(description="RL Benchmark by Yanjie Ze.")

# general setting
parser.add_argument('-b', '--benchmark', choices=['DQN', 'DoubleDQN', 'DuelingDQN'], default='DoubleDQN', help="Names of benchmarks you select.")
parser.add_argument('--env', choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0','Pendulum-v0'], default='CartPole-v1', help='Name of your envs.')
parser.add_argument('--epoch', type=int, default=500, help='Num of epochs.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device you use, cpu or cuda.')
parser.add_argument('--seed', default=0, type=float, help='Seed for randomness. Aim for reproducibility.')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for neural networks\' upate.')
parser.add_argument('--epsilon', default=0.01, type=float, help='Factor of epsilon-greedy exploration.')
parser.add_argument('--gamma', default=0.95, type=float, help='Discount factor of the Markov decision process.')
parser.add_argument('--hidden', default=128, type=int, help='Size of the hidden layer in neural networks.')
parser.add_argument('--save', default=True, type=bool, help='Whether to save the model\' s check point.')
parser.add_argument('--plot', default=True, type=bool, help='Whether to plot the result')

# DQN setting
parser.add_argument('--target_update', default=10, type=int, help='For DQN: updating target network in DQN per N iterations.')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for updating using replay buffer.')
parser.add_argument('--max_capacity', default=1000, type=int, help='Max capacity of the replay buffer.')

if __name__=='__main__':
    args = parser.parse_args()
    main(args)