from typing import Tuple
import abc
from helpers import MlpModel, combine_shape
import torch
from torch.distributions import Categorical, Normal, Distribution
import numpy as np
from gym.spaces import Space, Box, Discrete


class Critic(torch.nn.Module):
    def __init__(self, obs_size, hidden_sizes, activation, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.value_model = MlpModel(obs_size, hidden_sizes, 1, activation, None, device=device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor = self.value_model(obs)
        return torch.squeeze(value, -1)


class Actor(torch.nn.Module):
    @abc.abstractmethod
    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        pass

    @abc.abstractmethod
    def get_action_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, obs: torch.Tensor, action: torch.Tensor = None) -> Tuple[Distribution, torch.Tensor]:
        pi = self.get_distribution(obs)
        logp = None
        if action is not None:
            logp = self.get_action_log_prob(action)
        return pi, logp


class CategoricalActor(Actor):
    def __init__(self, obs_shape, act_shape, hidden_sizes=(64, 64), activation=torch.nn.Tanh(), output_activation=None, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.logits_model = MlpModel(obs_shape, hidden_sizes, act_shape, activation, output_activation, device=device)
        # noinspection PyTypeChecker
        self.pi: Categorical = None

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        logits: torch.Tensor = self.logits_model(obs)
        self.pi = Categorical(logits=logits)
        return self.pi

    def get_action_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.pi.log_prob(action)


class GaussianActor(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, hidden_sizes=(64, 64), activation=torch.nn.Tanh(), output_activation=None, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.mu_model = MlpModel(obs_shape, hidden_sizes, act_shape, activation, output_activation, device=device)
        log_std = -0.5 * np.ones(act_shape, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # noinspection PyTypeChecker
        self.pi: Normal = None

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        mu: torch.Tensor = self.mu_model(obs)
        std = torch.exp(self.log_std)
        self.pi = Normal(mu, std)
        return self.pi

    def get_action_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.pi.log_prob(action).sum(axis=-1)


class MlpActorCritic(torch.nn.Module):
    def __init__(self, observation_space: Space, action_space: Space, hidden_sizes=(64, 64), activation=torch.nn.Tanh(),
                 output_activation=None, device: torch.device = torch.device('cpu')):
        super(MlpActorCritic, self).__init__()

        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.pi = GaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation, device=device)
        elif isinstance(action_space, Discrete):
            self.pi = CategoricalActor(obs_dim, action_space.n, hidden_sizes, activation, device=device)

        self.value_function = Critic(obs_dim, hidden_sizes, activation, device=device)

    def step(self, obs: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            pi = self.pi.get_distribution(obs)
            a: torch.Tensor = pi.sample()
            logp_a = self.pi.get_action_log_prob(a)
            v: torch.Tensor = self.value_function(obs)
        return a.detach().cpu().numpy(), v.detach().cpu().numpy(), logp_a.detach().cpu().numpy()

    def act(self, obs: torch.Tensor):
        return self.step(obs)[0]