import argparse
import gym
import torch
import utils
import numpy as np
from colored import fg, bg, attr

# our benchmark
from DQN import DQN, ImgDQN
from DoubleDQN import DoubleDQN
from DuelingDQN import DuelingDQN
from Reinforce import Reinforce
from ActorCritic import ActorCritic

def main(args):

    print(args)

    # set seed (for reproducibility)
    seed = args.seed
    utils.setup_seed(seed)

    # set env
    env_name = args.env
    env = gym.make(env_name)
    env.reset()

    if not args.input_img:
        state_dim = env.observation_space.shape[0]
    else:
        state_dim = np.array([84, 84])

    if args.input_img:
        print ('%s Form of input: Image %s' % (fg('red') , attr(0)))
    else:
        print ('%s Form of input: State %s' % (fg('red') , attr(0)))
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
        if not args.input_img:
            agent = DQN(state_dim=state_dim, action_dim=action_dim,
                        hidden_dim=hidden_dim, learning_rate=lr,
                        gamma=gamma, epsilon=epsilon,
                        target_update=target_update,
                        max_capacity=args.max_capacity,
                        device=device).float()
        else:
            agent = ImgDQN(state_dim=state_dim, action_dim=action_dim,
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
    elif agent_name=='DuelingDQN':
        target_update = args.target_update
        agent = DuelingDQN(state_dim=state_dim, action_dim=action_dim,
                    hidden_dim=hidden_dim, learning_rate=lr,
                    gamma=gamma, epsilon=epsilon,
                    target_update=target_update,
                    max_capacity=args.max_capacity,
                    device=device).float()
    elif agent_name=='Reinforce':
        agent = Reinforce(state_dim=state_dim, 
                            action_dim=action_dim, 
                            hidden_dim=hidden_dim, 
                            learning_rate=lr, 
                            gamma=gamma,
                            device=device)
    elif agent_name=='ActorCritic':
        actor_lr = args.actor_lr
        critic_lr = args.critic_lr
        agent = ActorCritic(state_dim=state_dim, 
                            action_dim=action_dim, 
                            hidden_dim=hidden_dim, 
                            actor_lr=actor_lr,
                            critic_lr=critic_lr,
                            gamma=gamma,
                            device=device) 
    else:
        raise Exception('%s has not been implemented yet.'%agent_name)
    

    # running!
    epoch_num = args.epoch
    batch_size = args.batch_size
    reward_sum_list = []

    for epoch in range(epoch_num):

        cur_state = env.reset()
        cur_state = np.array(cur_state, dtype=np.float64)
        done = False

        reward_sum = 0 # record
        if agent_name in ['Reinforce', 'ActorCritic']:
            # record
            reward_list = []
            cur_state_list = []
            action_list = []
            next_state_list = []
            done_list = []

        while(not done):
            action = agent.act(cur_state)
            next_state,reward, done, _  = env.step(action)
            next_state = np.array(next_state, dtype=np.float64)
            
            # DQN, DoubleDQN, DuelingDQN
            if agent_name in ['DQN', 'DoubleDQN', 'DuelingDQN']:

                agent.record(cur_state, action, reward, next_state, done)
                if len(agent.replay_buffer) > batch_size:
                    transitions = agent.sample(batch_size)
                    agent.update(transitions)
                else:
                    continue
            # Reinforce
            elif agent_name in ['Reinforce', 'ActorCritic']  :
                action_list.append(action)
                reward_list.append(reward)
                cur_state_list.append(cur_state)
                next_state_list.append(next_state)
                done_list.append(done)
            # elif agent_name in ['ActorCritic']:
            #     transitions = {'states':cur_state, 'next_states':next_state, 'rewards':reward, 'dones':done,'actions':action}
            #     agent.update(transitions)
                
            cur_state = next_state # update state
            reward_sum += reward

            if done:
                break
        
        if agent_name in ['Reinforce', 'ActorCritic']:# update for Reinforce
            transitions = {'states':cur_state_list, 'actions':action_list, \
                        'rewards': reward_list, 'dones':done_list,\
                         'next_states':next_state_list}
            agent.update(transitions)
        
        if args.plot:
            reward_sum_list.append(reward_sum)
        print('epoch %u \'s reward sum: %f'%(epoch, reward_sum))

    if args.plot:
        utils.plot_one(model_name=agent_name, env_name=env_name, reward_list=reward_sum_list)


    if args.save:
        utils.save_model(model_name='%s_%s'%(agent_name,env_name), model=agent)

    


# parser
parser = argparse.ArgumentParser(description="RL Benchmark by Yanjie Ze.")


# general setting
benchmark_algorithms = ['DQN', 'DoubleDQN', 'DuelingDQN', 'Reinforce', 'ActorCritic','A2C','A3C']
parser.add_argument('-b', '--benchmark', choices=benchmark_algorithms, default='DQN', help="Names of benchmarks you select.")
parser.add_argument('--env', choices=['Pong-v0', 'Pong-ram-v0', 'CartPole-v1', 'CartPole-v0', 'Acrobot-v1', 'MountainCar-v0','Pendulum-v0'], default='CartPole-v1', help='Name of your envs.')
parser.add_argument('--epoch', type=int, default=1000, help='Num of epochs.')
parser.add_argument('--device', type=str, default='cuda:0', help='Device you use, cpu or cuda.')
parser.add_argument('--seed', default=0, type=float, help='Seed for randomness. Aim for reproducibility.')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for neural networks\' upate.')
parser.add_argument('--epsilon', default=0.01, type=float, help='Factor of epsilon-greedy exploration.')
parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor of the Markov decision process.')
parser.add_argument('--hidden', default=64, type=int, help='Size of the hidden layer in neural networks.')
parser.add_argument('--save', default=True, type=bool, help='Whether to save the model\' s check point.')
parser.add_argument('--plot', default=True, type=bool, help='Whether to plot the result')
parser.add_argument('--input_img', default=False, action='store_true')


# DQN and its family 's setting
parser.add_argument('--target_update', default=10, type=int, help='For DQN: updating target network in DQN per N iterations.')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for updating using replay buffer.')
parser.add_argument('--max_capacity', default=1000, type=int, help='Max capacity of the replay buffer.')

# Actor-Critic 's setting
parser.add_argument('--actor_lr', default=1e-3, type=float, help='Learning rate for the actor in Actor-Critic.')
parser.add_argument('--critic_lr', default=1e-2, type=float, help='Learning rate for the critic in Actor-Critic.')


args = parser.parse_args()

assert args.input_img+(args.env=='Pong-v0')==2 or args.input_img+(args.env=='Pong-v0')==0, 'Pong-v0 must use img as input, specify: --input_img'

if __name__=='__main__':
    main(args)