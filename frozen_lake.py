# Frozen Lake Q Learning

import gymnasium as gym
import numpy as np

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array

    learning_rate_a = 0.9 
    discount_factor_g = 0.9
    epsilon = 0.05
    rng = np.random.default_rng()   # random number generator

    for i in range(episodes):
        state = env.reset()[0] 
        terminated = False      
        truncated = False      

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() 
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action])

            state = new_state

    env.close()


if __name__ == '__main__':
    run(1000)