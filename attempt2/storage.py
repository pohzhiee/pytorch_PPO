from typing import Sequence, Dict
import numpy as np
import torch
from helpers import combine_shape, gae, discount_cumsum


class PPOBuffer:
    def __init__(self, obs_shape, act_shape, size: int = 10000, device:torch.device=torch.device('cpu')):
        self.buf_dict = {
            'obs': np.empty(combine_shape(size, obs_shape), dtype=np.float32),
            'act': np.empty(combine_shape(size, act_shape), dtype=np.float32),
            'rew': np.empty(size, dtype=np.float32),
            'done': np.empty(size, dtype=np.bool),
            'val': np.empty(size, dtype=np.float32),
            'logp': np.empty(size, dtype=np.float32),
            'adv': np.empty(size, dtype=np.float32),
            'cum_future_rew': np.empty(size, dtype=np.float32),
        }
        self.max_size = size
        self.ptr = 0
        self.episode_start_ptr = 0
        self.device = device

    def store(self, obs, act, rew, done, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.buf_dict['obs'][self.ptr] = obs
        self.buf_dict['act'][self.ptr] = act
        self.buf_dict['rew'][self.ptr] = rew
        self.buf_dict['done'][self.ptr] = done
        self.buf_dict['val'][self.ptr] = val
        self.buf_dict['logp'][self.ptr] = logp
        self.ptr += 1

    def finish_episode(self, last_rew, gamma, lambda_) -> None:
        """
        This function should be called to fill the buffers of the computed parameters such as advantage estimation at each step
        :param last_rew: reward of the last action taken
        :param gamma: Discount factor
        :param lambda_: Exponential averaging degree of weighting decrease (aka \alpha in wikipedia EMA), see GAE paper and Exponential moving average for more details
        :return: None
        """

        # Episode slice is basically just an alternative to using index, e.g. some_buf[start_ptr:ptr] == some_buf[episode_slice]
        episode_slice = slice(self.episode_start_ptr, self.ptr)
        # We add the last reward in because the initial state reward is useless
        episode_rewards = self.buf_dict['rew'][episode_slice]
        episode_rewards = np.append(episode_rewards, last_rew)
        # We also add the last reward in for the last state value function because we can
        # Maybe can consider predicting the last state value?
        episode_vals = self.buf_dict['val'][episode_slice]
        episode_vals = np.append(episode_vals, last_rew)
        episode_advantages = gae(gamma, lambda_, episode_rewards, episode_vals)
        # We ignore the first episode reward because the value at state 0 is the sum of reward of state 1 onwards
        # and has nothing to do with reward of state 0
        cumulative_future_rewards = discount_cumsum(episode_rewards[:-1], gamma)

        self.buf_dict['adv'][episode_slice] = episode_advantages
        self.buf_dict['cum_future_rew'][episode_slice] = cumulative_future_rewards

        self.episode_start_ptr = self.ptr

    def get_all(self) -> Dict[str, torch.Tensor]:
        data_slice = slice(0, self.ptr)
        return {k: torch.as_tensor(v[data_slice], dtype=torch.float32, device=self.device) for k, v in self.buf_dict.items()}

    def get_by_keys(self, keys: Sequence[str]):
        data_slice = slice(0, self.ptr)
        return {k: torch.as_tensor(v[data_slice], dtype=torch.float32, device=self.device) for k, v in self.buf_dict.items() if any(k == elem for elem in keys)}

    def clear(self):
        self.ptr = 0
        self.episode_start_ptr = 0
