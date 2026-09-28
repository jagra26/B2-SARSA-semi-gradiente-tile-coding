import time
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

random.seed(10)

from semi_gradient_sarsa import SemiGradientSarsa

env_ = gym.make("MountainCar-v0", render_mode=None)

sarsa = SemiGradientSarsa(env_)
sarsa.load_params()

env = gym.make("MountainCar-v0", render_mode='human')
state, info = env.reset(seed=10)
X, Y, Z = [state[0]], [state[1]], [0]

for i in range(300):
    action = sarsa.select_action(state, eps_greedy=False)
    print(action)
    next_state, reward, terminated, truncated, info = env.step(action)
    X.append(state[0])
    Y.append(state[1])
    Z.append(action)
    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.001)

    if terminated:
        time.sleep(1)
        break

    state = next_state

env.close()
