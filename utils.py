import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def setup_seed(seed):
    """
    Set up seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model_name, model):
    """
    Save model.
    """
    path = os.path.join('./checkpoint', '%s.pkl'%model_name)
    torch.save(model.state_dict(), path)


def plot_one(model_name, env_name, reward_list):
    """
    Plot result.
    """
    color_list = ['purple', 'red', 'cyan','blue', 'orange'  ,'pink','olive', 'green', 'gray']

    save_path = os.path.join('./result','%s_%s.png'%(model_name ,env_name))

    plt.style.use('fivethirtyeight')
    plt.figure()

    plt.plot(reward_list, color=color_list[0])
    plt.title('%s on %s'%(model_name ,env_name))
    plt.savefig(save_path)