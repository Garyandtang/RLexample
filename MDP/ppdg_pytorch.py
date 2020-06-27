import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import deque
# check cuda
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# replay buffer
# deque 类似列表(list)的容器，实现了在两端快速添加(append)和弹出(pop)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)

        )

if __name__ == '__main__':
    # set environment
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    # greedy exploration
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    # visualization
    # plt.plot([epsilon_by_frame(i) for i in range(10000)])
    # plt.show()


##################################################################################################
# import numpy as np
#
# x= np.array([[1,2,3],[4,5,6]])
#
# print(x)
#
# print(x.shape)
#
# print("-----------------------------")
#
# y= np.expand_dims(x, axis=0)
#
# print(y)
#
# print("y.shape: ", y.shape)
#
# print("-----------------------------")
#
# y= np.expand_dims(x, axis=1)
#
# print(y)
#
# print("y.shape: ", y.shape)
#
# print("-----------------------------")
#
# y= np.expand_dims(x, axis=2)
#
# print(y)
#
# print("y.shape: ", y.shape)

