import time
from typing import Union
from statistics import mean
import torch
from torch.distributions import normal
import numpy as np
import gym
import scipy.signal
from gym.envs.classic_control import PendulumEnv, CartPoleEnv
from ppo_mlp import MlpModel

torch.cuda.set_device(torch.cuda.current_device())
torch.set_default_tensor_type(torch.cuda.FloatTensor)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def gae_lambda(gamma, lambda_, rewards: Union[np.ndarray, torch.Tensor], predicted_values: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    deltas = rewards[:-1] + gamma * predicted_values[1:] - predicted_values[:-1]
    return discount_cumsum(deltas, gamma * lambda_)


class MlpActorCritic(torch.nn.Module):
    def __init__(self, obs_size, action_size, hidden_sizes=(64, 64), activation=torch.nn.Tanh(), output_activation=None):
        super(MlpActorCritic, self).__init__()
        self.action_shape = action_size
        # self.mu_model = MlpModel(obs_size, hidden_sizes, action_size, activation, output_activation).cuda()
        self.logits_model = MlpModel(obs_size, hidden_sizes, action_size, activation, output_activation).cuda()
        self.value_function_model = MlpModel(obs_size, hidden_sizes, 1, activation, None).cuda()
        self.EPS = 1e-8

    def gaussian_likelihood(self, x, mu, log_std):
        # pre_sum = log(f(x)), where f(x) is the pdf of gaussian distribution
        pre_sum = -0.5 * (((x - mu) / (torch.exp(log_std) + self.EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return pre_sum
        # return torch.sum(pre_sum, dim=1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor = None):
        # assert obs.ndimension() == 2, f'Expected observation to have dimension of 2, actual: {obs.ndimension()}'

        '''
        mu = self.mu_model(obs)
        log_std = torch.from_numpy(-0.5*np.ones(self.action_shape)).cuda()
        std = torch.exp(log_std)
        normal_dist = normal.Normal(0, std)
        normal_sample = normal_dist.sample(mu.shape)
        # normal_sample = np.random.normal(0, std, mu.shape)
        pi = mu + normal_sample * std
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)
        '''

        logits: torch.Tensor = self.logits_model(obs)
        logp_all = logits.log_softmax(0)
        categorical = torch.distributions.categorical.Categorical(logits=logits)
        pi = categorical.sample()  # policy selected action indices
        pi_indices_view = pi.unsqueeze(-1)
        logp_pi = logp_all.gather(-1, pi_indices_view.long())
        value = self.value_function_model(obs)

        if action is not None:
            assert action.ndimension() == 1, f'Expected action to have dimension of 1, actual: {action.ndimension()}'
            a_indices_view = action.view(-1, 1)
            logp = logp_all.gather(1, a_indices_view.long())
        else:
            logp = None
        '''
        if action is not None:
            assert action.ndimension() == 2, f'Expected action to have dimension of 2, actual: {action.ndimension()}'
            logp = self.gaussian_likelihood(action, mu, log_std).float()
        else:
            logp = None
        '''
        return pi, logp, logp_pi, value


class PPOBuffer:
    def __init__(self, obs_size, act_size, size: int = 10000):
        self.obs_buf = np.zeros([size] + list(obs_size), dtype=np.float32)
        self.act_buf = np.zeros([size] + list(act_size), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.empty(size, dtype=np.bool)
        self.val_buf = np.zeros(size, dtype=torch.FloatTensor)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=torch.FloatTensor)
        self.cum_future_rew_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.ptr = 0

    def store(self, obs, act, rew, done, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def store_adv_and_fut_rew(self, adv: np.ndarray, cumulative_future_reward: np.ndarray, start_ptr):
        # We assume that advantage and cumulative future reward are both 1D array
        assert len(adv) == len(cumulative_future_reward), 'Length of advantage and cumulative future reward must be the same'
        self.adv_buf[start_ptr:start_ptr + len(adv)] = adv
        self.cum_future_rew_buf[start_ptr:start_ptr + len(cumulative_future_reward)] = cumulative_future_reward

    def get_pointer(self) -> int:
        return self.ptr

    def get(self, start_ptr: int = 0):
        # Get the values up until that point in the buffer
        assert start_ptr < self.ptr, f'Start pointer must be smaller than current buffer pointer, ' \
                                     f'given start pointer: {start_ptr}, current buffer pointer: {self.ptr}'
        obs_buf = self.obs_buf[start_ptr:self.ptr]
        act_buf = self.act_buf[start_ptr:self.ptr]
        rew_buf = self.rew_buf[start_ptr:self.ptr]
        done_buf = self.done_buf[start_ptr:self.ptr]
        val_buf = self.val_buf[start_ptr:self.ptr]
        logp_buf = self.logp_buf[start_ptr:self.ptr]
        adv_buf = self.adv_buf[start_ptr:self.ptr]
        cum_future_rew_buf = self.cum_future_rew_buf[start_ptr:self.ptr]

        return [obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf, adv_buf, cum_future_rew_buf]

    def get_tuples(self):
        return [(a, b, c, d, e, f) for a, b, c, d, e, f in zip(*self.get())]

    def clear(self):
        # We only have to reset the pointer since we can't access the other elements without the pointer being incremented
        self.ptr = 0


class PPO:
    def __init__(self, gamma=0.99, lambda_=0.95, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=10, train_v_iters=80,
                 steps_per_epoch=5000):
        self.env: CartPoleEnv = gym.make('CartPole-v0')
        self.buffer: PPOBuffer = PPOBuffer(self.env.observation_space.shape, self.env.action_space.shape)
        if isinstance(self.env.action_space, gym.spaces.Box):
            action_size = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            action_size = self.env.action_space.n
        else:
            raise NotImplementedError(f'Action space not belonging to Box or Discrete not supported yet, is: {type(self.env.action_space)}')
        if isinstance(self.env.observation_space, gym.spaces.Box):
            obs_size = self.env.observation_space.shape[0]
        else:
            raise NotImplementedError('Observation space not belonging to Box not supported yet')
        self.policy = MlpActorCritic(obs_size, action_size)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr  # Policy learning rate
        self.vf_lr = vf_lr  # Value function learning rate
        self.train_pi_iters = 50
        self.train_v_iters = train_v_iters
        self.steps_per_epoch = steps_per_epoch
        self.episode_count = 0
        self.episode_reward = 0

    def run(self):
        obs_, rew_, done_ = self.env.reset(), 0, False
        prev_ptr = self.buffer.get_pointer()
        cumulative_reward_buffer = []
        for i in range(self.steps_per_epoch):
            obs_ = obs_.squeeze()
            pi, _, logp_pi, val = self.policy(torch.from_numpy(obs_).cuda().float())
            action = pi.cpu().detach().numpy()

            self.buffer.store(obs_, action, rew_, done_, val, logp_pi.cpu().detach().numpy())

            obs_, rew_, done_, info = self.env.step(action)
            self.episode_reward += rew_

            terminal = done_ or i == self.steps_per_epoch - 1
            if terminal:
                # Do buffer stuff, mainly calculating advantage and future rewards for each episode
                _, _, episode_reward_buffer, _, episode_val_buffer, _, _, _ = self.buffer.get(prev_ptr)
                episode_reward_buffer = np.append(episode_reward_buffer, rew_)
                # We just append the real to the estimation, since it is one step only and probably have no big impact
                episode_val_buffer = np.append(episode_val_buffer, rew_)

                advantage = gae_lambda(self.gamma, self.lambda_, episode_reward_buffer, episode_val_buffer)
                # We ignore the first episode reward because the value at state 0 is the sum of reward of state 1 onwards
                # and has nothing to do with reward of state 0
                rewards_to_go = discount_cumsum(episode_reward_buffer[1:], self.gamma)
                self.buffer.store_adv_and_fut_rew(advantage, rewards_to_go, start_ptr=prev_ptr)
                prev_ptr = self.buffer.get_pointer()

                # print(f'Episode {self.episode_count}: {self.episode_reward}')
                cumulative_reward_buffer.append(self.episode_reward)
                obs_, rew_, done_ = self.env.reset(), 0, False
                self.episode_count += 1
                self.episode_reward = 0

        print(f'Min: {min(cumulative_reward_buffer)}')
        print(f'Max: {max(cumulative_reward_buffer)}')
        print(f'Mean: {mean(cumulative_reward_buffer)}')

    def update(self):
        obs, act, rew, done, old_val, old_logp, adv, cum_future_rew = self.buffer.get()
        self.buffer.clear()
        for _ in range(self.train_pi_iters):
            # we ignore the last done for splitting because even if it's done it doesn't matter, as there will be nothing else continuing after it
            pi, logp, logp_pi, value = self.policy(torch.from_numpy(obs).cuda().float(), torch.from_numpy(act).cuda().float())
            # TODO: investigate why this gae lambda uses old value rather than new value for spinup
            # I think they are the same though because the network didn't change
            # Yea new and old are the same
            adv_tensor = torch.stack(adv.tolist())
            old_logp_tensor = torch.from_numpy(old_logp).cuda().float()
            ratio = torch.exp(logp - old_logp_tensor)
            min_adv = torch.where(adv_tensor > 0, (1 + self.clip_ratio) * adv_tensor, (1 - self.clip_ratio) * adv_tensor)
            pi_loss = torch.mean(torch.min(ratio * adv_tensor, min_adv))
            # v_loss = torch.mean((torch.from_numpy(cum_future_rew).cuda().float() - value) ** 2)
            mse_loss = torch.nn.MSELoss().cuda()
            v_loss = mse_loss(torch.from_numpy(cum_future_rew).cuda().float(), value.squeeze())
            #
            approx_kl = torch.mean(old_logp_tensor - logp)
            approx_ent = torch.mean(-logp)
            # # TODO: Get percentage of advantages clipped here
            #
            # # Training
            optim_pi = torch.optim.Adam(self.policy.logits_model.parameters(), lr=self.pi_lr)
            pi_loss.backward(retain_graph=True)
            optim_pi.step()
            optim_v = torch.optim.Adam(self.policy.value_function_model.parameters(), lr=self.vf_lr)
            v_loss.backward(retain_graph=True)
            optim_v.step()

        print(f'Approx kl: {approx_kl}')
        print(f'Approx entropy: {approx_ent}')
        print(f'V loss: {v_loss}')
        print(f'Pi loss: {pi_loss}')

a = PPO()
for _ in range(100):
    a.run()
    a.update()
