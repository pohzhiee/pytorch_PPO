import math
import torch
from torch.nn import Tanh
from torch.utils.tensorboard import SummaryWriter
import gym
from gym.envs.classic_control import PendulumEnv
import numpy as np

print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

EPS = 1e-8

def gaussian_likelihood(x, mu, log_std):
    # pre_sum = log(f(x)), where f(x) is the pdf of gaussian distribution
    pre_sum = -0.5 * (((x-mu)/(torch.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    print(pre_sum)
    return torch.sum(pre_sum)

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(self.hidden_size, hidden_size2)
        self.fc3 = torch.nn.Linear(self.hidden_size2, output_size)

        self.fc1.weight.data.fill_(0.1)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.weight.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.weight.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, x):
        hidden = self.fc1(x)
        tanh = self.tanh(hidden)
        output1 = self.fc2(tanh)
        tanh2 = self.tanh(output1)
        output = self.fc3(tanh2)
        return output


class GaussianPolicy:
    def __init__(self, obs, action, hidden_sizes=None, activation=None, output_activation=None):
        model = Model(4, 64, 64, action.shape[0])
        # obs_tensor = torch.from_numpy(obs).float()
        # action_tensor = torch.from_numpy(action).float()
        mu = model(obs)
        log_std = torch.from_numpy(-0.5*np.ones(action.shape))
        std = torch.exp(log_std)
        print(f'Std: {std}')
        normal_sample = np.random.normal(0, std, mu.shape)
        print(f'Normal sample: {normal_sample}')
        pi = mu + torch.from_numpy(normal_sample) * std
        logp = gaussian_likelihood(action, mu, log_std)
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        print(f'Mu: {mu}')
        print(f'Pi: {pi}')
        print(f'Logp: {logp}')
        print(f'Logp_pi: {logp_pi}')


writer = SummaryWriter('runs/test1')
model = Model(4, 64, 64, 1)
obs_tensor = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.0])).float()
np.random.seed(1)
action_tensor = torch.from_numpy(np.random.rand(1))
print(f'Action tensor: {action_tensor}')
policy = GaussianPolicy(obs_tensor, action_tensor)
env: PendulumEnv = gym.make('Pendulum-v0')
print(env.action_space.shape)
print(env.observation_space.shape)

# class PPO:
#     def __init__(self):
#         self.clip_ratio = 0.2
#         self.episodes_per_update = 5
#         self.train_pi_iters = 80
#         self.train_v_iters = 80
#         self.gamma = 0.99
#         self.pi_lr = 3e-4
#         self.vf_lr = 1e-3
#         self.lam = 0.97
#         self.target_kl = 0.01
#         self.save_freq = 10
#         self.env: PendulumEnv = gym.make('Pendulum-v0')
#         self.obs_buffer = []
#         self.reward_buffer = []
#         self.done_buffer = []
#
#
#     def run(self):
#         random_action = self.env.action_space.sample()
#         obs, reward, done, info = self.env.step(random_action)
#         self.obs_buffer.append(obs)
#         self.reward_buffer.append(reward)
#         self.done_buffer.append(done)
#
#
#     def advantage_model(self, ):
#         actor_dimension =