import torch
import numpy as np
import gym
from gym.envs.classic_control import PendulumEnv
from ppo_mlp import MlpModel


class MlpGaussianPolicy(torch.nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_sizes=(64,64), activation=torch.nn.Tanh(), output_activation=None):
        super(MlpGaussianPolicy, self).__init__()

        self.mu_model = MlpModel(obs_size, hidden_sizes, action_size, activation, output_activation)
        self.EPS = 1e-8

    def gaussian_likelihood(self, x, mu, log_std):
        # pre_sum = log(f(x)), where f(x) is the pdf of gaussian distribution
        pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + self.EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        print(pre_sum)
        return torch.sum(pre_sum)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        assert(obs.ndimension() == 2, f'Expected observation to have dimension of 2, actual: {obs.ndimension()}')
        assert(action.ndimension() == 2, f'Expected action to have dimension of 2, actual: {action.ndimension()}')
        mu = self.mu_model(obs)
        log_std = torch.from_numpy(-0.5*np.ones(action.shape))
        std = torch.exp(log_std)
        normal_sample = np.random.normal(0, std, mu.shape)
        pi = mu + torch.from_numpy(normal_sample) * std
        logp = self.gaussian_likelihood(action, mu, log_std)
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)

        return pi, logp, logp_pi

class PPOBuffer:
    def __init__(self, obs_size, act_size, size: int = 10000):
        self.obs_buf = np.zeros([size] + list(obs_size), dtype=np.float32)
        self.act_buf = np.zeros([size] + list(act_size), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.ptr = 0

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def get(self):
        # Get the values up until that point in the buffer
        obs_buf = self.obs_buf[:self.ptr]
        act_buf = self.act_buf[:self.ptr]
        rew_buf = self.rew_buf[:self.ptr]
        val_buf = self.val_buf[:self.ptr]
        logp_buf = self.logp_buf[:self.ptr]

        return [obs_buf, act_buf, rew_buf, val_buf, logp_buf]

    def get_tuples_list(self):
        return [(a, b, c, d, e) for a, b, c, d, e in zip(*self.get())]

    def clear(self):
        # We only have to reset the pointer since we can't access the other elements without the pointer being incremented
        self.ptr = 0


policy = MlpGaussianPolicy(4, 1)

obs_tensor = torch.from_numpy(np.array([1.0, 1.0, 1.0, 1.0])).float()
np.random.seed(1)
action_tensor = torch.from_numpy(np.random.rand(1))
pi, logp, logp_pi = policy(obs_tensor, action_tensor)
# env: PendulumEnv = gym.make('Pendulum-v0')
# buffer: PPOBuffer = PPOBuffer(env.observation_space.shape, env.action_space.shape)
# obs_, rew_, done_ = env.reset()
# buffer.store(obs_, rew_, done_, 0, logp=1)
# for i in range(300):
#     action = env.action_space.sample()
#     obs_, rew_, done_, info = env.step(action)
#     # env.render()
#     buffer.store(obs_, rew_, done_, 0, 1)
#     if done_:
#         env.reset()
#
# stuff = buffer.get_tuples_list()
# print('hello')