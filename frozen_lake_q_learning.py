# Frozen Lake Q Learning

import gymnasium as gym
import numpy as np
import pickle

def run(episodes, is_training=True, render=False):

    # Initializes the Environment
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)

    if(is_training):
        # Create a Q-Learning Table
        q = np.zeros((env.observation_space.n, env.action_space.n)) # 64 x 4
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # Define Parameters
    learning_rate = 0.9 
    discount_factor = 0.9 
    epsilon = 1         
    epsilon_decay_rate = 0.00001        
    random_number = np.random.default_rng()  


    for i in range(episodes):
        # Starting values
        state = env.reset()[0]  
        terminated = False      # True when falls or reaches goal
        truncated = False       # If actions are greater than 200

        while(not terminated and not truncated):
            # Use Epsilon-Decay Exploration
            if is_training and random_number.random() < epsilon:
                # Random Action
                action = env.action_space.sample() 
            else:
                action = np.argmax(q[state,:])

            next_state,reward,terminated,truncated,info = env.step(action)

            if reward == 0 and terminated == True:
                reward = -10
            elif reward == 1:
                reward = 100
            reward += -0.1

            if is_training:
                q[state,action] = q[state,action] + learning_rate * (reward + discount_factor * np.max(q[next_state,:]) - q[state,action])

            state = next_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate = 0.0001

    

    env.close()

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(10, is_training=False, render=True)