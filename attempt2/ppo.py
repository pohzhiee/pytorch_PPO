from typing import Tuple
from statistics import mean, stdev
import torch
from torch.optim import Adam
from torch.distributions import Distribution
import gym
import numpy as np
from actor_critic import MlpActorCritic
from storage import PPOBuffer

# TODO: V loss feels problematic, should be around 400-500 but why is mine 1300-1500??? what?
class PPO:
    def __init__(self, env: gym.Env, gamma=0.99, lambda_=0.97, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
                 steps_per_epoch=4000, device: torch.device = torch.device("cpu")):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.steps_per_epoch = steps_per_epoch
        self.device = device

        self.env = env
        self.buffer = PPOBuffer(env.observation_space.shape, env.action_space.shape, device=device)
        self.actor_critic = MlpActorCritic(self.env.observation_space, self.env.action_space, device=device)

        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.actor_critic.value_function.parameters(), lr=vf_lr)

        self.episode_count = 0
        self.episode_reward = 0

    def compute_loss_pi(self, obs: torch.Tensor, act: torch.Tensor, adv: torch.Tensor, logp_old: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pi, logp = self.actor_critic.pi(obs, act)
        pi: Distribution
        logp: torch.Tensor

        ratio = torch.exp(logp - logp_old)
        # stdev_adv = torch.std(adv)
        # mean_adv = torch.mean(adv)
        # normalised_adv = (adv-mean_adv)/stdev_adv
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        pi_loss = -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = torch.mean(logp_old - logp)
        approx_ent = torch.mean(-logp)
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean()

        return pi_loss, approx_kl, approx_ent, clipfrac

    def compute_loss_v(self, obs: torch.Tensor, cum_future_rew: torch.Tensor) -> torch.Tensor:
        mse_loss = torch.nn.MSELoss().to(self.device)
        value: torch.Tensor = self.actor_critic.value_function(obs)
        return mse_loss(value, cum_future_rew)

    def run(self):
        obs, rew, done = self.env.reset(), 0, False
        cumulative_reward_buffer = []
        for i in range(self.steps_per_epoch):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, val, logp = self.actor_critic.step(obs_tensor)

            next_obs, rew, done, info = self.env.step(action)

            self.buffer.store(obs, action, rew, done, val, logp)
            self.episode_reward += rew

            obs = next_obs

            # timeout = ep_len == max_ep_len
            # terminal = d
            epoch_ended = i == self.steps_per_epoch - 1

            if done or epoch_ended:
                # Do buffer stuff, mainly calculating advantage and future rewards for each episode, which is done in the finish_episode function

                if epoch_ended:
                    _, v, _ = self.actor_critic.step(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
                else:
                    v = 0

                self.buffer.finish_episode(v, self.gamma, self.lambda_)

                if done:
                    cumulative_reward_buffer.append(self.episode_reward)
                obs, rew, done = self.env.reset(), 0, False
                self.episode_count += 1
                self.episode_reward = 0

        print(f'Min: {min(cumulative_reward_buffer)}')
        print(f'Max: {max(cumulative_reward_buffer)}')
        print(f'Mean: {mean(cumulative_reward_buffer)}')

    def update(self):
        data_dict = self.buffer.get_all()
        pi_loss = None
        v_loss = None
        approx_ent = None
        approx_kl = None

        stdev_adv = torch.std(data_dict['adv'])
        mean_adv = torch.mean(data_dict['adv'])
        normalised_adv = (data_dict['adv']-mean_adv)/stdev_adv

        # pi_loss_old, _, _, _ = self.compute_loss_pi(data_dict['obs'], data_dict['act'], normalised_adv, data_dict['logp'])

        for _ in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            pi_loss, approx_kl, approx_ent, clipfrac = self.compute_loss_pi(data_dict['obs'], data_dict['act'], normalised_adv, data_dict['logp'])
            if approx_kl > 1.5 * 0.01:
                print("Stopping early...")
                break
            pi_loss.backward()
            self.pi_optimizer.step()

        for _ in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            v_loss = self.compute_loss_v(data_dict['obs'], data_dict['cum_future_rew'])
            v_loss.backward()
            self.vf_optimizer.step()

        self.buffer.clear()
        print(f'Approx kl: {approx_kl}')
        print(f'Approx entropy: {approx_ent}')
        print(f'V loss: {v_loss}')
        print(f'Pi loss: {pi_loss}')
        print(f'Clip frac: {clipfrac}')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    env = gym.make('CartPole-v1')
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    ppo = PPO(env, device=device)
    for i in range(60):
        print(f'----------------Epoch {i}------------------')
        ppo.run()
        ppo.update()