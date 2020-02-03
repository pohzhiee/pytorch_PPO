import numpy
import gym
from gym.envs.classic_control import CartPoleEnv


if __name__ == '__main__':
    env: CartPoleEnv = gym.make('CartPole-v0')
    env.seed(0)
    env.reset()
    numpy.random.seed(0)
    ep_rew = 0
    ep_rew_list = []
    for _ in range(200):
        act = numpy.random.binomial(1, 0.5)
        obs, rew, done, i = env.step(act)
        ep_rew += rew
        if done:
            ep_rew_list.append(ep_rew)
            ep_rew = 0
            env.reset()
    print(ep_rew_list)


