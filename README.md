# RL-Benchmark
An RL Benchmark. Easy to use, experiment, visualize, compare results, and extend.

# Supported Algorithms
- [ ] Model-free
    - [x] Value-based
        - [x] DQN
        - [x] Double DQN
        - [x] Dueling DQN
    - [ ] Policy-based
        - [x] Reinforce
        - [x] Actor-Critic
        - [ ] A2C
        - [ ] A3C
        - [ ] TRPO
        - [ ] ACKTR
        - [ ] PPO
        - [ ] DDPG
        - [ ] TD3
        - [ ] SAC
- [ ] Model-based



# Usage
Use `python run.py -h` to see the available parameters and get help.

To run a **demo**, simply run `python run.py`.

# Benchmark Results
## Value-based
![](./result/DQN_CartPole-v1.png)

![](./result/DoubleDQN_CartPole-v1.png)

![](./result/DuelingDQN_CartPole-v1.png)

## Policy-based
![](./result/Reinforce_CartPole-v1.png)

![](./result/ActorCritic_CartPole-v1.png)

# Implementation Details and Tricks
## DQN
1. Replay buffer: use `deque`.
2. Target network: hard update (load state dict each N iterations).
3. Only one hidden layer.
4. DQN's update utilizes `gather` in pytorch.
5. Data type: `torch.float` and `np.float`.
6. Fixed epsilon.
7. Target update=100 is worse than Target update=10.

Param:
```
Namespace(batch_size=64, benchmark='DQN', device='cuda:0', env='CartPole-v1', epoch=500, epsilon=0.01, gamma=0.95, hidden=128, lr=0.002, max_capacity=1000, plot=True, save=True, seed=0, target_update=10)
```
## Double DQN
1. Just change the way we computer Q target: $$r + Q_{target}(s, a_{target}) -> r + Q_{target}(s, a_{origin})$$

Param:
```
Namespace(batch_size=64, benchmark='DoubleDQN', device='cuda:0', env='CartPole-v1', epoch=500, epsilon=0.01, gamma=0.95, hidden=128, lr=0.001, max_capacity=1000, plot=True, save=True, seed=0, target_update=10)
```

## Dueling DQN
1. Change the network (add advantage function layer and value function layer).

Param:
```
Namespace(batch_size=64, benchmark='DuelingDQN', device='cuda:0', env='CartPole-v1', epoch=500, epsilon=0.01, gamma=0.95, hidden=128, lr=0.001, max_capacity=1000, plot=True, save=True, seed=0, target_update=10)
```

## Reinforce
1. Use `action_dist = torch.distributions.Categorical(probs)` to take action.

Param:
```
Namespace(batch_size=64, benchmark='Reinforce', device='cuda:0', env='CartPole-v1', epoch=1000, epsilon=0.01, gamma=0.98, hidden=128, lr=0.001, max_capacity=1000, plot=True, save=True, seed=0, target_update=10)
```
## Actor-Critic
1. Use TD error as the factor of the objective in policy gradient.
2. Use `detach()` to cut off the back propagation. (very important!)
3. Two-timescale update. `1e-3` for the actor, `1e-2` for the critic. (effect the performance very much)

Param:
```
Namespace(actor_lr=0.001, batch_size=64, benchmark='ActorCritic', critic_lr=0.01, device='cuda:0', env='CartPole-v1', epoch=1000, epsilon=0.01, gamma=0.99, hidden=128, lr=0.001, max_capacity=1000, plot=True, save=True, seed=0, target_update=10)
```

# Reference
1. [动手学强化学习](https://hrl.boyuai.com/)
