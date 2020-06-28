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


# class DQN(nn.Module):
#     def __init__(self, num_inputs, num_actions):
#         super(DQN, self).__init__()  # same as xxxxx
#
#         self.layers = nn.Sequential(
#             nn.Linear(env.observation_space.shape[0], 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, env.action_space.n)
#         )
#     def forward(self, x):
#         return self.layers(x)
#
#     def act(self, state, epsilon):
#         if random.random() > epsilon:
#             state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
#             q_value = self.forward(state)
#             action = q_value.max(1)[1].data[0]
#         else:
#             action = random.randrange(env.action_space.n)  # to explore the env
#         return action

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)
        return action

# Synchronize current policy net and target net
def update_target(current_model, traget_model):
    traget_model.load_state_dict(current_model.state_dict())


# Computing Temporal Difference Loss
# def compute_td_loss(batch_size, gamma):
#     state, action, reward, next_state, done = replay_buffer.sample(batch_size)
#
#     state = Variable(torch.FloatTensor(np.float32(state)))  # with data(tensor); grad (with same size as data but not Tensor); grad_fn
#     next_state = Variable(torch.FloatTensor(np.float32(next_state)))
#     action = Variable(torch.LongTensor(action))
#     reward = Variable(torch.FloatTensor(reward))
#     done = Variable(torch.FloatTensor(done))
#
#     q_values = current_model(state)
#     next_q_values = current_model(next_state)
#     next_q_state_values = target_model(next_state)
#
#     q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
#     next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
#     expected_q_value = reward + gamma * next_q_value * (1 - done)
#     # expected_q_value = reward + 0.99 * next_q_value * (1 - done)
#     loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     return loss


def compute_td_loss(batch_size, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = current_model(state)
    next_q_values = current_model(next_state)
    next_q_state_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    # set environment
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    # greedy exploration
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    # # visualization
    # plt.plot([epsilon_by_frame(i) for i in range(10000)])
    # plt.show()

    current_model = DQN(env.observation_space.shape[0], env.action_space.n)
    target_model = DQN(env.observation_space.shape[0], env.action_space.n)

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model = target_model.cuda()

    optimizer = optim.Adam(current_model.parameters())

    replay_buffer = ReplayBuffer(1000)
    update_target(current_model, target_model)

    num_frames = 10000
    batch_size = 32
    gamma = 0.99

    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon)

        next_state, reward, done, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size, gamma)
            losses.append(loss.item())

        if frame_idx % 200 == 0:
            plot(frame_idx, all_rewards, losses)

        if frame_idx % 100 == 0:
            update_target(current_model, target_model)

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

